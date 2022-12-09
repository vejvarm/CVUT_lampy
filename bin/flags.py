from os import curdir


class FLAGS:
    naccs = 6
    nlamps = 1  # for new X_li.npy files use 1, for old X.npy use 3
    naccs_per_lamp = 2

    # LAMPS_GRID = ["l1", "l2", "l3"]  #
    # BIN_SIZE_GRID = [(64, )]  #
    # THRESHOLD_GRID = [(0.01, 0.1)]  #
    # PERIOD_GRID = [1, 7, 14, 30]  #
    # VAR_SCALED_GRID = [True]  #

    TIMES_FACTOR = 1

    LAMPS_GRID = ["l2", ]
    BIN_SIZE_GRID = [(4, 8, 16), (64, )]
    THRESHOLD_GRID = [(0.05, ), (0.2, )]
    PERIOD_GRID = [1, 7]
    VAR_SCALED_GRID = [True, ]

    preproc_default = {
        "fs": 1000,  # Hz
        "ns_per_hz": 1,
        "freq_range": (0, 256),
        "tdf_order": 5,
        "tdf_ranges": ((95, 105), ),
        "use_autocorr": True,
        "noise_f_rem": (1,),
        "noise_df_rem": (1,),
        "mov_filt_size": 1,
        "rem_neg": True,
    }

    mat_field_names = {
        "Accs": r"Acc\d[a-zA-Z]?",
        "fs": r"Frekvence(Mereneho)?Signalu",
        "WindDirection": "WindDirection",
        "WindSpeed": "WindSpeed",
    }

    data_root = curdir
    raw_folder = "raw"
    preprocessed_folder = "preprocessed"
    image_save_folder = "results/images"

    setting = "_no_autocorr"

    paths = {
        "training": {"folder": "training",
                     "dataset": [""],
                     "period": f"2months{setting}"
                     },
        "validation": {"folder": "validation",
                       "dataset": [f"unbroken{setting}", f"broken{setting}"],
                       "period": ""
                       },
        "serialized": {"folder": f"serialized{setting}",  # TODO: Change if needed ("serialized_full")
                       "dataset": [""],
                       "period": ""
                       }
    }

    serialized = {"unbroken": 107,  # 167,
                  "broken": 436}  # 93 TODO: Change if needed

    lamps = {
        "l1": ("X_l1.npy", "y_l1.npy"),
        "l2": ("X_l2.npy", "y_l2.npy"),
        "l3": ("X_l3.npy", "y_l3.npy"),
    }

    preprocessing = {
        "use_autocorr": True,
        "rem_neg": False  # works better than True
    }

    PSNR_csv_setup = {
        "name": "PSNR.csv",
        "name2": "PSNRda.csv",
        "columns": ["episode", "signal_amp_max", "noise_amp_max", "train_PSNR (dB)", "test_PSNR (dB)"],
        "sep": ";",
        "decimal": ",",
        "index": False,
        "line_terminator": "\n"
    }
