import utility.util as util
import prep.image_snippet as snippet
import prep.downscale as ds
import os
import datetime

# hard coded directories
top_folder = "train-data"
div2k_dir = "C:/Users/John/Documents/BA/DIV2K Dataset/DIV2K_train_HR"
# top_folder = "val-data"
# div2k_dir = "C:/Users/John/Documents/BA/DIV2K Dataset/DIV2K_valid_HR"
origin_dir = f"{top_folder}/original"
control_dir = f"{top_folder}/control"  # no shift directory
half_dir = f"{top_folder}/half_shift"
quarter_dir = f"{top_folder}/quarter_shift"
random0_dir = f"{top_folder}/random0_shift"


def div2k_name(number):
    return os.path.join(div2k_dir, f"{number:04d}.png")


def extract_origin_img(image, padded):
    origin_image = snippet.extract_random_snippet(image)
    origin_name = os.path.join(origin_dir, "original-" + padded)
    util.save_array_image(origin_image, origin_name)
    return origin_image


def create_control_img(image, padded_name):
    control_img = ds.batch_shift_downscale(image, (0, 0))
    control_name = os.path.join(control_dir, "control-" + padded_name)
    util.save_array_image(control_img, control_name)


def create_half_img(image, padded_name):
    half_img = ds.batch_shift_downscale(image, (0, 0.5))
    half_name = os.path.join(half_dir, "half-" + padded_name)
    util.save_array_image(half_img, half_name)


def create_quarter_img(image, padded_name):
    quarter_img = ds.batch_shift_downscale(image, (0, 0.25, 0.5, 0.75))
    quarter_name = os.path.join(quarter_dir, "quarter-" + padded_name)
    util.save_array_image(quarter_img, quarter_name)


def create_random0_img(image, padded_name):
    random0_img = ds.batch_shift_downscale(image, snippet.random0_shifts())
    random0_name = os.path.join(random0_dir, "random0-" + padded_name)
    util.save_array_image(random0_img, random0_name)


def create_dataset(start, stop):
    print(f"{datetime.datetime.now().time()}: Start")
    for i in range(start, stop):
        img = util.load_img_ndarray(div2k_name(i))

        samples_per_img = 10
        for j in range(samples_per_img):
            image_number = (i - 1) * samples_per_img + j
            img_name = f"{image_number:05d}.png"

            origin_img = extract_origin_img(img, img_name)

            create_control_img(origin_img, img_name)
            create_half_img(origin_img, img_name)
            create_quarter_img(origin_img, img_name)
            create_random0_img(origin_img, img_name)

        print(f"{datetime.datetime.now().time()}: {i} done")
    print(f"{datetime.datetime.now().time()}: Finished")


if __name__ == "__main__":
    # create all necessary directories
    util.create_dir(origin_dir)
    util.create_dir(control_dir)
    util.create_dir(half_dir)
    util.create_dir(quarter_dir)
    util.create_dir(random0_dir)

    create_dataset(1, 801)  # train_HR
    # create_dataset(801, 901)  # valid_HR
