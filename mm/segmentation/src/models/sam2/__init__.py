from mm.utils.git import install_by_clone

try:
    import sam2 
    print("CANNOT import sam2, need to install first")
except:
    install_by_clone(url="https://github.com/facebookresearch/sam2.git", dir_name='sam2')  