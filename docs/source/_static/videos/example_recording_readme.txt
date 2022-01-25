single_row_horizontal.mp4
Preview frame 3112 (incl.) to 3920 (incl.)

single_row_vertical.mp4
Preview frame 400 (incl.) to 1248 (incl.) of multiple rows experiment one rows

double_row_vertical.mp4
Preview frame 1386 (incl.) to 2175 (incl.) of multiple rows experiment two rows



Encoded with:
ffmpeg -framerate 8 -start_number 1386 -i frame_%06d.jpg -c:v libx264 -pix_fmt yuv420p output.mp4
