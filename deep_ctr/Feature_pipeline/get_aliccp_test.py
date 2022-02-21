import re
import numpy as np

Common_Fileds   = {'101':'1','121':'2','122':'3','124':'4','125':'5','126':'6','127':'7','128':'8','129':'9','205':'10','301':'11'}
UMH_Fileds      = {'109_14':('u_cat','12'),'110_14':('u_shop','13'),'127_14':('u_brand','14'),'150_14':('u_int','15')}      #user multi-hot feature
Ad_Fileds       = {'206':('a_cat','16'),'207':('a_shop','17'),'210':('a_int','18'),'216':('a_brand','19')}  

text = '40362692,0,0,216:9342395:1.0 301:9351665:1.0 205:7702673:1.0 206:8317829:1.0 207:8967741:1.0 508:9356012:2.30259 210:9059239:1.0 210:9042796:1.0 210:9076972:1.0 210:9103884:1.0 210:9063064:1.0 127_14:3529789:2.3979 127_14:3806412:2.70805'
fields = text.strip().split(',')
y = [float(fields[1])]
z = [float(fields[2])]
feature = {
    "y": y,
    "z": z
}

splits = re.split('[ :]', fields[3])
ffv = np.reshape(splits, (-1, 3))
# 2 不需要特殊处理的特征
feat_ids = np.array([])
for f, def_id in Common_Fileds.items():
    if f in ffv[:, 0]:
        mask = np.array(f == ffv[:, 0])
        feat_ids = np.append(feat_ids, ffv[mask, 1])
    else:
        feat_ids = np.append(feat_ids, def_id)
feature.update({"feat_ids": feat_ids.tolist()})

# 3 特殊字段单独处理
for f, (fname, def_id) in UMH_Fileds.items():
    if f in ffv[:, 0]:
        mask = np.array(f == ffv[:, 0])
        feat_ids = ffv[mask, 1]
        feat_vals = ffv[mask, 2]
    else:
        feat_ids = np.array([def_id])
        feat_vals = np.array([1.0])
    feature.update({fname+"ids": feat_ids.tolist(),
                    fname+"vals": feat_vals.tolist()})

for f, (fname, def_id) in Ad_Fileds.items():
    if f in ffv[:, 0]:
        mask = np.array(f == ffv[:, 0])
        feat_ids = ffv[mask, 1]
    else:
        feat_ids = np.array([def_id])
    feature.update({fname+"ids": feat_ids.tolist()})

print(feature)
