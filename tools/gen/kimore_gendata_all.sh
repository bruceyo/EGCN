#!/bin/bash

python ./tools/gen/kimore_gendata_cv_cs.py --joint_feature position --out_folder ./data/KiMoRe/cv_cs/ang
python ./tools/gen/kimore_gendata_cv_cs.py --joint_feature angle --out_folder ./data/KiMoRe/cv_cs/xyz
python ./tools/gen/kimore_gendata_cv_cs.py --joint_feature both --out_folder ./data/KiMoRe/cv_cs/xyzang
python ./tools/gen/kimore_gendata_cv_rd.py --joint_feature position --out_folder ./data/KiMoRe/cv_rd/ang
python ./tools/gen/kimore_gendata_cv_rd.py --joint_feature angle --out_folder ./data/KiMoRe/cv_rd/xyz
python ./tools/gen/kimore_gendata_cv_rd.py --joint_feature both --out_folder ./data/KiMoRe/cv_rd/xyzang
