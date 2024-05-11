import os
from pathlib import Path

p = Path('mlruns')

#201 e 251
for f in p.rglob('*siamese_opt_no_cloud*.*'):
    print(f"Rename {f.name} to {f.name.replace('opt_', 'opt_no_prevmap_')}")
    f.rename(f.parent / f.name.replace('opt_', 'opt_no_prevmap_'))

#202 e 252 siamese_sar_no_prevmap_avg2
for f in p.rglob('*siamese_sar_avg2*.*'):
    print(f"Rename {f.name} to {f.name.replace('sar_', 'sar_no_prevmap_')}")
    f.rename(f.parent / f.name.replace('sar_', 'sar_no_prevmap_'))
 
#203 e 253 siamese_sar_no_prevmap_single2
for f in p.rglob('*siamese_sar_single2*.*'):
    print(f"Rename {f.name} to {f.name.replace('sar_', 'sar_no_prevmap_')}")
    f.rename(f.parent / f.name.replace('sar_', 'sar_no_prevmap_'))
 
#204 e 254 siamese_opt_no_cloud
for f in p.rglob('*siamese_opt_prevmap_no_cloud*.*'):
    print(f"Rename {f.name} to {f.name.replace('opt_prevmap_', 'opt_')}")
    f.rename(f.parent / f.name.replace('opt_prevmap_', 'opt_'))
 
#205 e 255 siamese_sar_avg2
for f in p.rglob('*siamese_sar_prevmap_avg2*.*'):
    print(f"Rename {f.name} to {f.name.replace('sar_prevmap_', 'sar_')}")
    f.rename(f.parent / f.name.replace('sar_prevmap_', 'sar_'))
    
#206 e 256 siamese_sar_single2
for f in p.rglob('*siamese_sar_prevmap_single2*.*'):
    print(f"Rename {f.name} to {f.name.replace('sar_prevmap_', 'sar_')}")
    f.rename(f.parent / f.name.replace('sar_prevmap_', 'sar_'))