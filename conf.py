""" default channel last and learning rate annealing used default"""

config_map = {"data_root"        : "/home/anish/anish/datas/DUT/",
              "backbone"         : "res50v2",
       	      "epochs"           : 25,
       	      "warmup"           : 5,
       	      "learning_rate"    : 1.21e-4,
       	      "dim"              : 416, 
              "crop_patch_interval" :[310,400],
       	      "batch"            : 6,
       	      "tbl"              : "./logs"
       	      }




#model catlogue ["res50","res50v2","res101v2","dense121"]
