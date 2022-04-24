# -*- coding: utf-8 -*-
import click
import os
import sqlite3
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas

from src.data.utils import (
    build_demographic_table,
    build_diagnoses_table,
    build_lab_table,
    build_prescriptions_table,
    build_procedures_table,
    show_progress,
)


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final mimic iii dataset")
    data_dir = input_filepath
    # Path to the generated mimic.db. No need to update.
    out_dir = output_filepath

    conn = sqlite3.connect(os.path.join(out_dir, "mimic_all.db"))
    id2name_path = os.path.join(data_dir, "id2name.csv")
    mimic_files_path = os.path.join(data_dir, "files")

    build_demographic_table(mimic_files_path, id2name_path, out_dir, conn)
    build_diagnoses_table(mimic_files_path, out_dir, conn)
    build_procedures_table(mimic_files_path, out_dir, conn)
    build_prescriptions_table(mimic_files_path, out_dir, conn)
    build_lab_table(mimic_files_path, out_dir, conn)

    print("Begin sampling ...")
    # DEMOGRAPHIC
    print("Processing DEMOGRAPHIC")
    conn = sqlite3.connect(os.path.join(out_dir, "mimic.db"))
    data_demo = pandas.read_csv(os.path.join(out_dir, "DEMOGRAPHIC.csv"))
    data_demo_sample = data_demo.sample(100, random_state=0)
    data_demo_sample.to_sql("DEMOGRAPHIC", conn, if_exists="replace", index=False)
    sampled_id = data_demo_sample["HADM_ID"].values

    # DIAGNOSES
    print("Processing DIAGNOSES")
    data_input = pandas.read_csv(os.path.join(out_dir, "DIAGNOSES.csv"))
    data_filter = []
    cnt = 0
    for itm in sampled_id:
        msg = "HADM_ID==" + str(itm)
        data_filter.append(data_input.query(msg))
        cnt += 1
        show_progress(cnt, len(sampled_id))
    data_out = pandas.concat(data_filter, ignore_index=True)
    data_out.to_sql("DIAGNOSES", conn, if_exists="replace", index=False)

    # PROCEDURES
    print("Processing PROCEDURES")
    data_input = pandas.read_csv(os.path.join(out_dir, "PROCEDURES.csv"))
    data_filter = []
    cnt = 0
    for itm in sampled_id:
        msg = "HADM_ID==" + str(itm)
        data_filter.append(data_input.query(msg))
        cnt += 1
        show_progress(cnt, len(sampled_id))
    data_out = pandas.concat(data_filter, ignore_index=True)
    data_out.to_sql("PROCEDURES", conn, if_exists="replace", index=False)

    # PRESCRIPTIONS
    print("Processing PRESCRIPTIONS")
    data_input = pandas.read_csv(os.path.join(out_dir, "PRESCRIPTIONS.csv"))
    data_filter = []
    cnt = 0
    for itm in sampled_id:
        msg = "HADM_ID==" + str(itm)
        data_filter.append(data_input.query(msg))
        cnt += 1
        show_progress(cnt, len(sampled_id))
    data_out = pandas.concat(data_filter, ignore_index=True)
    data_out.to_sql("PRESCRIPTIONS", conn, if_exists="replace", index=False)

    # LAB
    print("Processing LAB")
    data_input = pandas.read_csv(os.path.join(out_dir, "LAB.csv"))
    data_filter = []
    cnt = 0
    for itm in sampled_id:
        msg = "HADM_ID==" + str(itm)
        data_filter.append(data_input.query(msg))
        cnt += 1
        show_progress(cnt, len(sampled_id))
    data_out = pandas.concat(data_filter, ignore_index=True)
    data_out.to_sql("LAB", conn, if_exists="replace", index=False)
    print("Done!")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
