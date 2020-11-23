#!/bin/bash

python ./tools/gen/ui_prmd_gendata_cv_cs.py --joint_feature position --out_folder ./data/UI_PRMD/cv_cs/ang
python ./tools/gen/ui_prmd_gendata_cv_cs.py --joint_feature angle --out_folder ./data/UI_PRMD/cv_cs/xyz
python ./tools/gen/ui_prmd_gendata_cv_cs.py --joint_feature both --out_folder ./data/UI_PRMD/cv_cs/xyzang
python ./tools/gen/ui_prmd_gendata_cv_rd.py --joint_feature position --out_folder ./data/UI_PRMD/cv_rd/ang
python ./tools/gen/ui_prmd_gendata_cv_rd.py --joint_feature angle --out_folder ./data/UI_PRMD/cv_rd/xyz
python ./tools/gen/ui_prmd_gendata_cv_rd.py --joint_feature both --out_folder ./data/UI_PRMD/cv_rd/xyzang
