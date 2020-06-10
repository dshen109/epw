"""
Convert NOAA files to EPW files.
"""

from collections import OrderedDict
import os

import numpy as np
import pandas as pd
import pint
import yaml

UREG = pint.UnitRegistry()


with open("epw_schema.yaml", "r") as f:
    epw_schema = yaml.load(f)
    EPW_SCHEMA = OrderedDict()
    for item in epw_schema["fields"]:
        k = list(item.keys())[0]
        EPW_SCHEMA[k] = item[k]


with open("lcd_to_epw.yaml", "r") as f:
    LCD_TO_EPW_SCHEMA = yaml.load(f)

RAIN = 919999999
SNOW = 999199999
DRY = 999999999

TZ_LOCAL = "America/Chicago"


class EPW:

    def __init__(self, frame):
        """ Initialize from a dataframe. """
        self.frame = frame

    @classmethod
    def from_dtindex_df(cls, frame):
        """ Load from a datatimeindex'd DataFrame. """
        pass

    def write(self, filename):
        """ Write to a file.  """
        output = self.frame.copy()
        field_ordered = [k for k in EPW_SCHEMA.keys()]
        for field, params in EPW_SCHEMA.items():
            missing_default = params.get("missing")

            if field not in output.columns:
                output[field] = missing_default
                print("Filled {} column with missing value of {}"
                      .format(field, missing_default))
            else:
                nans = output[field].isna()
                numnan = nans.sum()
                output.loc[nans, field] = missing_default
                if numnan:
                    print(
                        "Filled {} nan values in {} with missing value of {}"
                        .format(numnan, field, missing_default))
                minimum = params.get("min")
                maximum = params.get("max")
                if minimum is not None:
                    under = (
                        (output[field] < minimum) & ~output[field].isna())
                    numunder = under.sum()
                    output.loc[under, field] = missing_default
                    if numunder:
                        print(
                            "Filled {} under values in {} with "
                            "missing value of {}"
                            .format(numunder, field, missing_default))
                if maximum is not None:
                    over = (output[field] > maximum) & ~output[field].isna()
                    numover = over.sum()
                    output.loc[over, field] = missing_default
                    if numover:
                        print(
                            "Filled {} over values in {} with "
                            "missing value of {}"
                            .format(numover, field, missing_default))
        output["Horizontal Infrared Radiation Intensity"] = \
            self.calc_horizontal_infrared_radiation(
                output["Dry Bulb Temperature"] + 273.15,
                output["Dew Point Temperature"] + 273.15,
                output["Opaque Sky Cover"])

        for field, params in EPW_SCHEMA.items():
            output[field] = output[field].astype(params.get("type", "int"))
            if params.get("type") == "float":
                output[field] = round(output[field], params.get("decimals", 1))
            output[field] = output[field].astype(str)

        output[field_ordered].to_csv(filename, index=False, header=False)

    @staticmethod
    def calc_horizontal_infrared_radiation(
            drybulb, dewpoint, opaque_sky_cover):
        """
        :param drybulb: Drybulb temperautre in Kelvin
        :param dewpoint: Dewpoint temperature in Kelvin
        :param opaque_sky_cover: Sky cover in tenths
        """
        emissivity = (
            (0.787 + 0.764 * np.log(dewpoint / 273)) * (
                1 + 0.0224 * opaque_sky_cover +
                0.0035 * opaque_sky_cover ** 2 +
                0.00028 * opaque_sky_cover ** 3))
        boltzmann = 5.6697e-8
        return emissivity * boltzmann * drybulb


class NOAAData:
    """
    Container for NOAA Local Climatalogical Data.
    """

    def __init__(self, frame):
        """
        Initialize object.
        """
        self.frame = frame

    @classmethod
    def load_csv(cls, filename):
        """ Load a NOAA file from csv. Assumes timestamps are in UTC. """
        weather = pd.read_csv(
            filename, index_col=1, parse_dates=True)
        weather = weather.tz_localize("UTC")
        weather.dropna(how="all", axis=1, inplace=True)
        return cls(weather)

    @classmethod
    def load_pickle(cls, filename):
        """ Load a NOAA file from a pickle. """
        data = pd.read_pickle(filename)
        return cls(data)

    def to_epw(self, interval="5T", start="", end=""):
        """
        Convert into an EPW file. Start and end are in local timestamps.
        """
        frame = self.frame.copy().resample(interval).first()

        frame = frame.tz_convert(TZ_LOCAL)

        if start:
            frame = frame.loc[start:]
        if end:
            frame = frame.loc[:end]

        output = pd.DataFrame()
        # Convert columns.
        for epw_col, details in LCD_TO_EPW_SCHEMA.items():
            print("Handling column: {}".format(epw_col))
            numeric = True
            # Convert rain / snow to appropriate weather code.
            if epw_col == "Present Weather Codes":
                numeric = False
                weather_code = frame[details["source"]]
                weather_code.fillna(" ", inplace=True)
                raining = (
                    weather_code.str.contains("DZ") |
                    weather_code.str.contains("RA"))
                snowing = weather_code.str.contains("SN")
                neither = ~snowing & ~raining

                weather_code.loc[raining] = RAIN
                weather_code.loc[snowing] = SNOW
                weather_code.loc[neither] = DRY
                output[epw_col] = weather_code.values
            # No source column in the weather CSV, fill with default value.
            elif not details.get("source"):
                output[epw_col] = details["default"]
            # Convert the sky cover to a number out of 10
            elif details["source"] == "HourlySkyConditions":
                sky = frame[details["source"]]
                sky = pd.to_numeric(sky.str[5], errors="coerce") / 8 * 10
                output[epw_col] = sky.values
            else:
                source = pd.to_numeric(
                    frame[details["source"]].copy(), errors='coerce').values
                if details.get("from"):
                    unit_from = UREG(details["from"])
                    unit_to = UREG(details["to"])
                    source = (source * unit_from).to(unit_to).magnitude
                output[epw_col] = source
            # Interpolate up to 60 minutes of data.
            interp_limit = 60 * 60 / pd.to_timedelta(interval).total_seconds()
            interp_limit = int(interp_limit)
            if numeric:
                output[epw_col] = output[epw_col].interpolate(
                    limit=interp_limit)
            else:
                output[epw_col] = output[epw_col].interpolate(
                    "pad", limit=interp_limit)

            print(
                "Average value of {}: {}"
                .format(epw_col, output[epw_col].mean()))
        output = round(output, 3)
        output["Year"] = frame.index.year
        output["Month"] = frame.index.month
        output["Day"] = frame.index.day
        output["Hour"] = frame.index.hour
        output["Minute"] = frame.index.minute

        return EPW(output)


def prepend_lines_from_epw(file_name, epw_file):
    """ Prepend EPW header lines for design days into a file. """
    # define name of temporary dummy file
    dummy_file = file_name + '.bak'
    # open original file in read mode and dummy file in write mode
    with open(file_name, 'r') as read_obj, open(epw_file, 'r') as epw_obj, \
            open(dummy_file, 'w') as write_obj:
        # Write given line to the dummy file
        for i, epw_line in enumerate(epw_obj):
            write_obj.write(epw_line)
            if i == 7:
                break
        # Read lines from original file one by one and append them to the dummy file
        for line in read_obj:
            write_obj.write(line)
    # remove original file
    os.remove(file_name)
    # Rename dummy file as the original file
    os.rename(dummy_file, file_name)


if __name__ == "__main__":
    epw_out = "chicago_5m_complete.epw"
    freq = "5T"

    noaa = NOAAData.load_csv("chicago_ohare_2019_complete.csv")
    epw = noaa.to_epw(freq, "2019-01-01 00:00", "2019-12-31 23:59:59")
    epw.write(epw_out)
    prepend_lines_from_epw(epw_out, "chicago_midway.epw")
