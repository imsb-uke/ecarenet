import pandas as pd


def create_img_and_label_series(csv_filename, data_generation_config):
    """
    read a csv file which contains all information about the dataset:
    this includes the filepath, the label, some additional information (maybe about censoring or patient age)
    further, some patients can be dropped, which should not be used further (for example all patients with unknown '-1'
             censoring status)
    :param csv_filename: complete path to csv
    :param data_generation_config: config in with
                                    annotation_column: string, which column to find (main) label information (time to bcr)
                                    additional_columns: None or
                                                        list of dict('string' internal_name: ['string' column_in_csv, 'string' - label_or_-])
                                   drop_cases: None or list of list(value_to_drop, internal_name_of_variable] to drop
    :return pd Series image paths,
            pd Series labels in original format,
            dict of pd Series with additional column information

    """

    # read the csv and the desired columns
    f = pd.read_csv(csv_filename)

    images = f['img_path']
    labels = f[data_generation_config['annotation_column']]
    if data_generation_config['additional_columns'] is not None:
        additional_columns = {}
        add_labels = {k: d[k] for d in data_generation_config['additional_columns'] for k in d}
        for al in add_labels:
            if add_labels[al] is not None:
                additional_columns[al] = f[add_labels[al][0]]
    else:
        additional_columns = []

    # remove unwanted classes
    if data_generation_config['drop_cases'] is not None:
        for idx, d in enumerate(data_generation_config['drop_cases']):
            if isinstance(d, list):
                column_remove = d[1]
                value_remove = d[0]
            else:
                column_remove = data_generation_config['annotation_column']
                value_remove = d
            if column_remove != data_generation_config['annotation_column']:
                # value to remove is in additional labels
                value_remove = type(additional_columns[column_remove][0])(value_remove)
                images = images[(additional_columns[column_remove] != value_remove) &
                                (~pd.isna(additional_columns[column_remove]))]
                labels = labels[(additional_columns[column_remove] != value_remove) &
                                (~pd.isna(additional_columns[column_remove]))]

                for al in additional_columns:
                    if al != column_remove:
                        additional_columns[al] = additional_columns[al][(additional_columns[column_remove] != value_remove)
                                                                      & (~pd.isna(additional_columns[column_remove]))
                                                                      ]
                        additional_columns[al] = additional_columns[al].reset_index(drop=True)

                additional_columns[column_remove] = additional_columns[column_remove][
                    (additional_columns[column_remove] != value_remove) & (~pd.isna(additional_columns[column_remove]))
                                                                                    ]
                additional_columns[column_remove] = additional_columns[column_remove].reset_index(drop=True)
                images = images.reset_index(drop=True)
                labels = labels.reset_index(drop=True)

            else:
                # value to remove is in the class label
                value_remove = type(labels)(value_remove)

                for al in additional_columns:
                    additional_columns[al] = additional_columns[al][~labels.isin(value_remove) & (~pd.isna(labels))]
                    additional_columns[al] = additional_columns[al].reset_index(drop=True)

                images = images[~labels.isin(value_remove) & (~pd.isna(labels))]
                labels = labels[~labels.isin(value_remove) & (~pd.isna(labels))]

                images = images.reset_index(drop=True)
                labels = labels.reset_index(drop=True)

    return images, labels, additional_columns
