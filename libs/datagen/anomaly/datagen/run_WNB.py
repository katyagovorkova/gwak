import os
import numpy as np
import time
from anomaly.datagen.injection_2d import main_WNB

wnb=False
if wnb:
    path = "/home/ryan.raikman/s23/data/WNB_pols/"
    pre_tag = "wnb_"
    post_tag = "_0.01_40_400"
    save_tag = "WNB"
    main_WNB(f"/home/ryan.raikman/s23/data/WNB_strains/{post_tag[1:]}",
            "/home/ryan.raikman/s22/anomaly/data2/glitches/1239134846_1239140924/",
            path, pre_tag=pre_tag, post_tag=post_tag, save_tag=save_tag)
else:
    #SN
    author = "Powell_2021"
    path = f"/home/ryan.raikman/s23/data/SN_pols/{author}/"
    pre_tag = "SN_"
    post_tag = ""
    save_tag ="SN"
    main_WNB(f"/home/ryan.raikman/s23/data/SN_strains_long/{author}",
            "/home/ryan.raikman/s22/anomaly/data2/glitches/1239134846_1239140924/",
            path, pre_tag=pre_tag, post_tag=post_tag, save_tag=save_tag, segment_length=8)
