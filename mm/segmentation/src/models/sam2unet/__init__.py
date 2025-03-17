from mm.segmentation.src.models.sam2 import install_sam2

try:
    import sam2 
    print("CANNOT import sam2, need to install first")
except:
    install_sam2()
    
    
    