import matplotlib


def save_gif(images, path, duration=10):
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0,
    )


def plot_thumbnails(ax, images, positions, zoom=0.05):
    for image, position in zip(images, positions):
        im = matplotlib.offsetbox.OffsetImage(image, zoom=zoom)
        ab = matplotlib.offsetbox.AnnotationBbox(
            im, position, xycoords="data", frameon=False
        )
        ax.add_artist(ab)
