import zarr
y=zarr.open('/home/group1/26fall_aiclass/zjt/cira-diff/datasets/satcast/edm_GOES_ch13_test_dataset.zarr',mode='r')['output_images']
hits=[]
for i in range(1024):
    tb=y[i,0]*19.3297+279.0699        # 归一化值 -> 开尔文
    cold=float((tb<=235).mean())
    if cold>0.02:
        print(f"index={i}, cold_cloud_cover={cold}")
