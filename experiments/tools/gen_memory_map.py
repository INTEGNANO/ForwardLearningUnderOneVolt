row_offset = 15
col_offset = 0
width = 32
height = 32

mat_width = 32
mat_height = 4

step = 1
row_first = True

out = ""
for i in range(0, mat_width * mat_height):
    if row_first:
        row = int((step * i) // (width // step))
        col = int((step * i) % (width // step))
    else:
        col = (step * i) // (height // step)
        row = (step * i) % (height // step)

    if row >= height:
        raise ValueError(
            f"Software Weights do not fit in the Hardware Array (row ({row}) > height ({height}))"
        )
    if col >= width:
        raise ValueError(
            f"Software Weights do not fit in the Hardware Array (col ({col}) > width ({width}))"
        )

    out += f"{row_offset + row:02},{col_offset + col:02}"

    if i % mat_width == mat_width - 1:
        out += "\n"
    else:
        out += " "

with open("../data/memory_maps/tmp.dat", "a") as f:
    f.write(out)
