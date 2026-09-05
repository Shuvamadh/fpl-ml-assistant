"""Generate assets/fpl.ico - dark FPL-flavoured app icon. Re-runnable."""
from PIL import Image, ImageDraw, ImageFont
import os

BG_A = (23, 27, 48)      # deep navy
BG_B = (10, 12, 26)
ACCENT = (0, 255, 135)   # FPL green
MAGENTA = (233, 0, 82)   # FPL magenta

def find_font(size):
    for p in (r"C:\Windows\Fonts\segoeuib.ttf", r"C:\Windows\Fonts\arialbd.ttf",
              r"C:\Windows\Fonts\calibrib.ttf"):
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()

def draw(size):
    S = size * 4  # supersample
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    # rounded square background with vertical gradient
    grad = Image.new("RGBA", (S, S))
    gd = ImageDraw.Draw(grad)
    for y in range(S):
        t = y / (S - 1)
        gd.line([(0, y), (S, y)], fill=(
            int(BG_A[0] + (BG_B[0]-BG_A[0])*t),
            int(BG_A[1] + (BG_B[1]-BG_A[1])*t),
            int(BG_A[2] + (BG_B[2]-BG_A[2])*t), 255))
    mask = Image.new("L", (S, S), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, S-1, S-1], radius=int(S*0.22), fill=255)
    img.paste(grad, (0, 0), mask)
    d = ImageDraw.Draw(img)
    # accent underline bar
    d.rounded_rectangle([int(S*0.20), int(S*0.80), int(S*0.80), int(S*0.855)],
                        radius=int(S*0.03), fill=ACCENT)
    # magenta corner tick
    d.rounded_rectangle([int(S*0.20), int(S*0.145), int(S*0.44), int(S*0.185)],
                        radius=int(S*0.02), fill=MAGENTA)
    # bold "F"
    f = find_font(int(S*0.62))
    text = "F"
    bb = d.textbbox((0, 0), text, font=f)
    w, h = bb[2]-bb[0], bb[3]-bb[1]
    d.text((S/2 - w/2 - bb[0], S*0.47 - h/2 - bb[1]), text, font=f, fill=ACCENT)
    return img.resize((size, size), Image.LANCZOS)

sizes = [16, 32, 48, 64, 128, 256]
imgs = [draw(s) for s in sizes]
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fpl.ico")
imgs[-1].save(out, format="ICO", sizes=[(s, s) for s in sizes], append_images=imgs[:-1])
print("wrote", out, os.path.getsize(out), "bytes")
