"""A clearly synthetic, non-evaluation image for checking the agent wiring."""
from pathlib import Path
import sys
from PIL import Image, ImageDraw
path=Path(sys.argv[1])
path.parent.mkdir(parents=True,exist_ok=True)
image=Image.new('RGB',(512,512),(80,130,180))
draw=ImageDraw.Draw(image)
draw.ellipse((150,100,360,310),fill=(220,170,100))
draw.rectangle((190,300,320,450),fill=(60,70,120))
image.save(path)
print('Created synthetic debug image:',path)
