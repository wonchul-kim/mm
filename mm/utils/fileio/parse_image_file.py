import struct

def get_image_size(file_path):
    with open(file_path, "rb") as f:
        header = f.read(24)  # Read first 24 bytes (covers all formats)

        # BMP (first 2 bytes: "BM")
        if header[:2] == b'BM':  
            f.seek(18)  # BMP width & height are at offset 18
            width, height = struct.unpack("<ii", f.read(8))
            return width, abs(height)  # Height can be negative (top-down BMP)

        # PNG (first 8 bytes: PNG signature, IHDR at byte 16)
        elif header[:8] == b"\x89PNG\r\n\x1a\n":
            width, height = struct.unpack(">II", header[16:24])
            return width, height

        # JPEG (Marker format: SOI 0xFFD8, followed by markers)
        elif header[:2] == b"\xFF\xD8":
            f.seek(2)
            while True:
                marker, = struct.unpack(">H", f.read(2))  # Read marker
                size, = struct.unpack(">H", f.read(2))  # Read segment size
                if marker in {0xFFC0, 0xFFC2}:  # Start of Frame markers
                    f.seek(1, 1)  # Skip precision byte
                    height, width = struct.unpack(">HH", f.read(4))
                    return width, height
                f.seek(size - 2, 1)  # Skip non-SOF segments
        raise NotImplementedError(f"[ERROR] NOT Considered yet this case of image: {file_path}")
