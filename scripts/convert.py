"""
Convert NOAA files to EPW files.
"""

import os

import numpy as np
import pandas as pd
import pint
import yaml

UREG = pint.UnitRegistry()


with open("epw_schema.yaml", "r") as f:
    EPW_SCHEMA = yaml.load(f)

with open("lcd_to_epw.yaml", "r") as f:
    LCD_TO_EPW_SCHEMA = yaml.load(f)

RAIN = 919999999
SNOW = 999199999
DRY = 999999999


class EPW:

    def __init__(self, frame):
        """ Initialize from a dataframe. """
        self.frame = frame

    @classmethod
    def from_dtindex_df(cls, frame):
        """ Load from a datatimeindex'd DataFrame. """
        pass

    def write(self, filename):
        """ Write to a file. """
        output = self.frame.copy()
        field_ordered = [list(k.keys())[0] for k in EPW_SCHEMA["fields"]]
        for i, field in enumerate(field_ordered):
            missing_default = EPW_SCHEMA["fields"][i][field].get("missing")
            # if (field not in output.columns and
            #         field != "Horizontal Infrared Radiation Intensity"):
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
                minimum = EPW_SCHEMA["fields"][i][field].get("min")
                maximum = EPW_SCHEMA["fields"][i][field].get("max")
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

        output["Present Weather Codes"] = \
            output["Present Weather Codes"].astype(int)
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
        """ Load a NOAA file from csv. """
        weather = pd.read_csv(
            filename, index_col=1, parse_dates=True)
        weather.dropna(how="all", axis=1, inplace=True)
        return cls(weather)

    @classmethod
    def load_pickle(cls, filename):
        """ Load a NOAA file from a pickle. """
        data = pd.read_pickle(filename)
        return cls(data)

    def to_epw(self, interval="5T"):
        """
        Convert into an EPW file.
        """
        output = pd.DataFrame()
        frame = self.frame.copy().resample(interval).first()
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
            if numeric:
                output[epw_col] = output[epw_col].interpolate(limit=15)
            else:
                output[epw_col] = output[epw_col].interpolate("pad", limit=15)

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
    noaa = NOAAData.load_csv("chicago_ohare_2019.csv")
    epw = noaa.to_epw("60T")
    epw.write("test.epw")
    prepend_lines_from_epw("test.epw", "chicago_midway.epw")
