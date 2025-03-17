from mm.utils.git import install_by_clone

try:
    import sam2 
    print("CANNOT import sam2, need to install first")
except:
    install_by_clone(url="https://github.com/facebookresearch/sam2.git", dir_name='sam2')  
    print("SUCCESSFULLY INSTALLED sam2 !!!")
    
from .backbones import SAM2Encoder
from .decode_heads.sam2unet_head import SAM2UNetHead


__all__ = ['SAM2Encoder', 'SAM2UNetHead']