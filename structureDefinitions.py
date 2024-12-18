# images
images_dir = "images/"


# layouts
layouts_dir = "layouts/"
layout_ext = "-layout"

def layout_path(path):
    image_name = path.split("/")[-1].split(".")[0]
    return layouts_dir + image_name + layout_ext


# openCV window
window_width, window_height = 1400, 900