from time import sleep
from tqdm import tqdm

i = 0

data = list(range(100))
size = len(data)
pbar = tqdm(data)
j = 0
for i, item in enumerate(pbar):
    sleep(0.25)
    pbar.set_description("Current element %i/%i score: %f" % (i, size, j))
    j+=1