""" default channel last and learning rate annealing used default"""

config_map = {"data_root"        : "/home/anish/anish/GCPAnet/DUT_BENCH/",
              "backbone"         : "res50v2",
       	      "epochs"           : 30,
       	      "warmup"           : 5,
       	      "learning_rate"    : 1e-5,
       	      "dim"              : 416, 
              "crop_patch_interval" :[280,400],
       	      "batch"            : 8,
       	      "tbl"              : "./logs"
       	      }




#model catlogue ["res50","res50v2","res101v2","dense121"]
