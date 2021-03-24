class FLAGS:
    naccs = 6
    nlamps = 1  # for new X_li.npy files use 1, for old X.npy use 3
    naccs_per_lamp = 2

    preproc_default = {
        "fs": 1000,  # Hz
        "ns_per_hz": 10,
        "freq_range": (0, 500),
        "tdf_order": 5,
        "tdf_ranges": ((45, 55), (95, 105), (145, 155), (195, 205), (245, 255), (295, 305), (345, 355), (395, 405), (445, 455)),
        "use_autocorr": True,
        "noise_f_rem": (1,),
        "noise_df_rem": (1,),
        "mov_filt_size": 5,
        "rem_neg": True,
    }

    mat_field_names = {
        "Accs": r"Acc\d[a-zA-Z]?",
        "fs": r"Frekvence(Mereneho)?Signalu",
        "WindDirection": "WindDirection",
        "WindSpeed": "WindSpeed",
    }

    data_root = "b:/!Cloud/OneDrive - VŠCHT/CVUT_Lampy/data/"
    raw_folder = "raw"
    preprocessed_folder = "preprocessed"

    paths = {
        "training": {"folder": "training",
                     "dataset": [""],
                     "period": "2months"
                     },
        "validation": {"folder": "validace",
                       "dataset": ["neporuseno", "poruseno"],
                       "period": ""
                       },
        "serialized": {"folder": "serialized",
                       "dataset": [""],
                       "period": ""
                       }
    }

    serialized = {"unbroken": 107,
                  "broken": 93}  # TODO: Change if needed

    lamps = {
        "l1": ("X_l1.npy", "y_l1.npy"),
        "l2": ("X_l2.npy", "y_l2.npy"),
        "l3": ("X_l3.npy", "y_l3.npy"),
    }

    preprocessing = {
        "use_autocorr": True,
        "rem_neg": True
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
