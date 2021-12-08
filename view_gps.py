import os
import glob
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt


def main(work_dir):
    gps_file = os.path.join(work_dir, "splitted", "gps", "gps.csv")
    positions = np.genfromtxt(gps_file, delimiter=",")

    frame_dir = os.path.join(work_dir, "splitted", "preview")
    frame_files = sorted(glob.glob(os.path.join(frame_dir, "*")))

    # if limits is not None:
    #     positions = positions[limits[0]:limits[1], :]
    #     frame_files = frame_files[limits[0]:limits[1]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    ax1.plot(positions[:, 0], positions[:, 1], c="b", linewidth=1)
    sc = ax1.scatter(positions[:, 0], positions[:, 1], s=10, c="b")
    im = ax2.imshow(np.zeros((1, 1)), cmap="gray", aspect="auto")

    annot = ax1.annotate("", xy=(0,0), xytext=(20,20), textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)
    ax2.axis('off')
    ax1.axis('off')
    #ax1.set_xlabel("Longitude")
    #ax1.set_ylabel("Latitude")

    def update_annot(ind, max_items=1):
        pos = sc.get_offsets()[ind["ind"][0]]
        annot.xy = pos
        items = list(map(str, ind["ind"]))
        if max_items is not None and len(items) > max_items:
            text = "{}".format(",".join(items[:max_items] + ["..."]))
        else:
            text = "{}".format(",".join(items))
        annot.set_text(text)


    def hover(event):
        if event.inaxes == ax1:
            cont, ind = sc.contains(event)
            if cont:
                idx = ind["ind"][0]
                image = cv2.imread(frame_files[idx])
                im.set_data(image)
                update_annot(ind)
                annot.set_visible(True)
                fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)
    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Visualize video frames and corresponding GPS positions")
    parser.add_argument("workdir", type=str, help="workdir of the plant you want to visualize")
    #parser.add_argument("--limits", nargs=2, type=int, help="Index range (first frame, last frame) of frames to show. Omit to show all frames.")
    args = parser.parse_args()

    main(args.workdir)
