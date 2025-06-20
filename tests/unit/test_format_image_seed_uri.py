import unittest
from dgenerate.textprocessing import format_image_seed_uri


class TestFormatImageSeedURI(unittest.TestCase):

    def test_only_seed_image(self):
        result = format_image_seed_uri(seed_images="seed.png")
        self.assertEqual(result, "seed.png")

    def test_seed_and_inpaint_image(self):
        result = format_image_seed_uri(seed_images="seed.png", mask_images="inpaint.png")
        self.assertEqual(result, "seed.png;inpaint.png")

    def test_seed_inpaint_and_control_image(self):
        result = format_image_seed_uri(seed_images="seed.png", mask_images="inpaint.png", control_images="control.png")
        self.assertEqual(result, "seed.png;mask=inpaint.png;control=control.png")

    def test_seed_and_control_image(self):
        result = format_image_seed_uri(seed_images="seed.png", control_images="control.png")
        self.assertEqual(result, "seed.png;control=control.png")

    def test_resize_with_seed_image(self):
        result = format_image_seed_uri(seed_images="seed.png", resize="800x600")
        self.assertEqual(result, "seed.png;800x600")

    def test_resize_width_only_with_seed_image(self):
        result = format_image_seed_uri(seed_images="seed.png", resize="800")
        self.assertEqual(result, "seed.png;800")

    def test_resize_tuple_with_seed_image(self):
        result = format_image_seed_uri(seed_images="seed.png", resize=("800", "600"))
        self.assertEqual(result, "seed.png;800x600")

    def test_resize_mixed_tuple_with_seed_image(self):
        result = format_image_seed_uri(seed_images="seed.png", resize=(800, "600"))
        self.assertEqual(result, "seed.png;800x600")

    def test_aspect_false_with_seed_image(self):
        result = format_image_seed_uri(seed_images="seed.png", aspect=False)
        self.assertEqual(result, "seed.png;aspect=False")

    def test_frame_start_and_end_with_seed_image(self):
        result = format_image_seed_uri(seed_images="seed.png", frame_start=0, frame_end=10)
        self.assertEqual(result, "seed.png;frame-start=0;frame-end=10")

    def test_keyword_arguments_without_seed_or_control_image(self):
        result = format_image_seed_uri(seed_images="seed.png", frame_start=0)
        self.assertEqual(result, "seed.png;frame-start=0")

    def test_inpaint_image_without_seed_image(self):
        with self.assertRaises(ValueError):
            format_image_seed_uri(seed_images=None, mask_images="inpaint.png")

    def test_control_image_without_seed_or_inpaint_image(self):
        result = format_image_seed_uri(seed_images=None, control_images="control.png")
        self.assertEqual(result, "control.png")

    def test_all_arguments(self):
        result = format_image_seed_uri(
            seed_images="seed.png",
            mask_images="inpaint.png",
            control_images="control.png",
            resize="800x600",
            aspect=False,
            frame_start=0,
            frame_end=10
        )
        self.assertEqual(result,
                         "seed.png;mask=inpaint.png;control=control.png;resize=800x600;aspect=False;frame-start=0;frame-end=10")

    def test_invalid_resize(self):
        with self.assertRaises(ValueError):
            format_image_seed_uri(seed_images="seed.png", resize="invalid")

    def test_invalid_resize_non_digit(self):
        with self.assertRaises(ValueError):
            format_image_seed_uri(seed_images="seed.png", resize="800xabc")

    def test_invalid_resize_non_tuple(self):
        with self.assertRaises(ValueError):
            format_image_seed_uri(seed_images="seed.png", resize=("800", "abc"))

    def test_invalid_resize_with_special_characters(self):
        with self.assertRaises(ValueError):
            format_image_seed_uri(seed_images="seed.png", resize="800x600@")

    def test_invalid_resize_with_letters(self):
        with self.assertRaises(ValueError):
            format_image_seed_uri(seed_images="seed.png", resize="8a00x600")

    def test_none_values(self):
        result = format_image_seed_uri(seed_images="seed.png", mask_images=None, control_images=None, resize=None,
                                       aspect=True, frame_start=None, frame_end=None)
        self.assertEqual(result, "seed.png")

    def test_empty_strings(self):
        result = format_image_seed_uri(seed_images="seed.png", mask_images="", control_images="")
        self.assertEqual(result, "seed.png")

    def test_resize_empty_string(self):
        result = format_image_seed_uri(seed_images="seed.png", resize="")
        self.assertEqual(result, "seed.png")

        result = format_image_seed_uri(seed_images="seed.png", resize="\n\t\r")
        self.assertEqual(result, "seed.png")

        result = format_image_seed_uri(seed_images="seed.png", resize="  ")
        self.assertEqual(result, "seed.png")

    def test_frame_values(self):
        result = format_image_seed_uri(seed_images="seed.png", frame_start=0, frame_end=0)
        self.assertEqual(result, "seed.png;frame-start=0;frame-end=0")

        result = format_image_seed_uri(seed_images="seed.png", frame_start=6, frame_end=8)
        self.assertEqual(result, "seed.png;frame-start=6;frame-end=8")

    def test_aspect_false(self):
        result = format_image_seed_uri(seed_images="seed.png", aspect=False)
        self.assertEqual(result, "seed.png;aspect=False")

    def test_resize(self):
        result = format_image_seed_uri(seed_images="seed.png", resize="800x600")
        self.assertEqual(result, "seed.png;800x600")

    def test_all_keyword_arguments(self):
        result = format_image_seed_uri(seed_images="seed.png", resize="800x600", aspect=False, frame_start=0,
                                       frame_end=10)
        self.assertEqual(result, "seed.png;resize=800x600;aspect=False;frame-start=0;frame-end=10")

    def test_incorrect_resize_format(self):
        with self.assertRaises(ValueError):
            format_image_seed_uri(seed_images="seed.png", resize="800x600x400")

    def test_missing_seed_image_with_inpaint_and_control(self):
        with self.assertRaises(ValueError):
            format_image_seed_uri(seed_images=None, mask_images="inpaint.png", control_images="control.png")

    def test_frame_start_without_frame_end(self):
        result = format_image_seed_uri(seed_images="seed.png", frame_start=10)
        self.assertEqual(result, "seed.png;frame-start=10")

    def test_frame_end_without_frame_start(self):
        result = format_image_seed_uri(seed_images="seed.png", frame_end=10)
        self.assertEqual(result, "seed.png;frame-end=10")

    # New tests for adapter and floyd images
    def test_adapter_images_only(self):
        result = format_image_seed_uri(seed_images=None, adapter_images=["adapter1.png", "adapter2.png"])
        self.assertEqual(result, "adapter:adapter1.png + adapter2.png")

    def test_seed_with_adapter_images(self):
        result = format_image_seed_uri(seed_images="seed.png", adapter_images=["adapter1.png", "adapter2.png"])
        self.assertEqual(result, "seed.png;adapter=adapter1.png + adapter2.png")

    def test_floyd_image_with_seed(self):
        result = format_image_seed_uri(seed_images="seed.png", floyd_image="floyd.png")
        self.assertEqual(result, "seed.png;floyd=floyd.png")

    def test_floyd_image_with_inpaint(self):
        result = format_image_seed_uri(mask_images="inpaint.png", floyd_image="floyd.png")
        self.assertEqual(result, "inpaint.png;floyd=floyd.png")

    def test_floyd_and_control_image_conflict(self):
        with self.assertRaises(ValueError):
            format_image_seed_uri(seed_images="seed.png", floyd_image="floyd.png", control_images="control.png")

    def test_adapter_and_floyd_image_conflict(self):
        with self.assertRaises(ValueError):
            format_image_seed_uri(seed_images="seed.png", adapter_images=["adapter1.png"], floyd_image="floyd.png")

    def test_seed_with_adapter_and_mask(self):
        result = format_image_seed_uri(
            seed_images="seed.png",
            adapter_images=["adapter1.png", "adapter2.png"],
            mask_images="mask.png"
        )
        self.assertEqual(result, "seed.png;mask=mask.png;adapter=adapter1.png + adapter2.png")

    def test_seed_with_adapter_and_control(self):
        result = format_image_seed_uri(
            seed_images="seed.png",
            adapter_images=["adapter1.png", "adapter2.png"],
            control_images="control.png"
        )
        self.assertEqual(result, "seed.png;adapter=adapter1.png + adapter2.png;control=control.png")

    def test_seed_with_adapter_mask_and_control(self):
        result = format_image_seed_uri(
            seed_images="seed.png",
            adapter_images=["adapter1.png", "adapter2.png"],
            mask_images="mask.png",
            control_images="control.png"
        )
        self.assertEqual(result, "seed.png;mask=mask.png;adapter=adapter1.png + adapter2.png;control=control.png")

    def test_basic_seed_with_adapter(self):
        result = format_image_seed_uri(
            seed_images='image',
            adapter_images=['test']
        )
        self.assertEqual(result, "image;adapter=test")

    # New tests for list parameters
    def test_seed_images_as_list(self):
        result = format_image_seed_uri(seed_images=["seed1.png", "seed2.png"])
        self.assertEqual(result, "images:seed1.png, seed2.png")

    def test_mask_images_as_list(self):
        result = format_image_seed_uri(seed_images="seed.png", mask_images=["mask1.png"])
        self.assertEqual(result, "seed.png;mask1.png")

    def test_control_images_as_list(self):
        result = format_image_seed_uri(seed_images="seed.png", control_images=["control1.png", "control2.png"])
        self.assertEqual(result, "seed.png;control=control1.png, control2.png")
    
    def test_latents_as_list(self):
        result = format_image_seed_uri(latents=["latent1.pt", "latent2.pt"])
        self.assertEqual(result, "latents:latent1.pt, latent2.pt")
    
    def test_seed_images_and_mask_images_as_lists(self):
        result = format_image_seed_uri(
            seed_images=["seed1.png", "seed2.png"], 
            mask_images=["mask1.png", "mask2.png"]
        )
        self.assertEqual(result, "images:seed1.png, seed2.png;mask1.png, mask2.png")
    
    def test_seed_images_and_control_images_as_lists(self):
        result = format_image_seed_uri(
            seed_images=["seed1.png", "seed2.png"], 
            control_images=["control1.png", "control2.png"]
        )
        self.assertEqual(result, "images:seed1.png, seed2.png;control=control1.png, control2.png")
    
    def test_seed_mask_and_control_as_lists(self):
        result = format_image_seed_uri(
            seed_images=["seed1.png", "seed2.png"],
            mask_images=["mask1.png", "mask2.png"],
            control_images=["control1.png", "control2.png"]
        )
        self.assertEqual(result, "images:seed1.png, seed2.png;mask=mask1.png, mask2.png;control=control1.png, control2.png")
    
    def test_seed_mask_control_and_adapter_as_lists(self):
        result = format_image_seed_uri(
            seed_images=["seed1.png", "seed2.png"],
            mask_images=["mask1.png", "mask2.png"],
            control_images=["control1.png", "control2.png"],
            adapter_images=["adapter1.png", "adapter2.png"]
        )
        self.assertEqual(result, "images:seed1.png, seed2.png;mask=mask1.png, mask2.png;adapter=adapter1.png + adapter2.png;control=control1.png, control2.png")
    
    def test_latents_with_other_params(self):
        result = format_image_seed_uri(
            seed_images=["seed1.png", "seed2.png"],
            latents=["latent1.pt", "latent2.pt"]
        )
        self.assertEqual(result, "images:seed1.png, seed2.png;latents=latent1.pt, latent2.pt")
    
    def test_all_list_params_with_other_arguments(self):
        result = format_image_seed_uri(
            seed_images=["seed1.png", "seed2.png"],
            mask_images=["mask1.png", "mask2.png"],
            control_images=["control1.png", "control2.png"],
            adapter_images=["adapter1.png", "adapter2.png"],
            latents=["latent1.pt", "latent2.pt"],
            resize="800x600",
            aspect=False,
            frame_start=0,
            frame_end=10
        )
        self.assertEqual(result, "images:seed1.png, seed2.png;mask=mask1.png, mask2.png;latents=latent1.pt, latent2.pt;adapter=adapter1.png + adapter2.png;control=control1.png, control2.png;resize=800x600;aspect=False;frame-start=0;frame-end=10")
    
    def test_equal_number_of_seed_and_mask_images(self):
        result = format_image_seed_uri(
            seed_images=["seed1.png", "seed2.png"],
            mask_images=["mask1.png", "mask2.png"]
        )
        self.assertEqual(result, "images:seed1.png, seed2.png;mask1.png, mask2.png")
    
    def test_single_mask_for_multiple_seeds(self):
        result = format_image_seed_uri(
            seed_images=["seed1.png", "seed2.png", "seed3.png"],
            mask_images=["mask1.png"]
        )
        self.assertEqual(result, "images:seed1.png, seed2.png, seed3.png;mask1.png")
    
    def test_too_many_mask_images(self):
        with self.assertRaises(ValueError):
            format_image_seed_uri(
                seed_images=["seed1.png", "seed2.png"],
                mask_images=["mask1.png", "mask2.png", "mask3.png"]
            )
    
    def test_too_few_mask_images(self):
        with self.assertRaises(ValueError):
            format_image_seed_uri(
                seed_images=["seed1.png", "seed2.png", "seed3.png"],
                mask_images=["mask1.png", "mask2.png"]
            )
    
    def test_latents_only_as_list(self):
        result = format_image_seed_uri(latents=["latent1.pt", "latent2.pt"])
        self.assertEqual(result, "latents:latent1.pt, latent2.pt")
    
    def test_control_images_only_as_list(self):
        result = format_image_seed_uri(control_images=["control1.png", "control2.png"])
        self.assertEqual(result, "control1.png, control2.png")
    
    def test_adapter_images_only_as_list(self):
        result = format_image_seed_uri(adapter_images=["adapter1.png", "adapter2.png"])
        self.assertEqual(result, "adapter:adapter1.png + adapter2.png")
    
    def test_latents_with_resize_error(self):
        with self.assertRaises(ValueError):
            format_image_seed_uri(
                latents=["latent1.pt", "latent2.pt"],
                resize="800x600"
            )
    
    def test_latents_with_aspect_error(self):
        with self.assertRaises(ValueError):
            format_image_seed_uri(
                latents=["latent1.pt", "latent2.pt"],
                aspect=False
            )
    
    def test_latents_with_frame_error(self):
        with self.assertRaises(ValueError):
            format_image_seed_uri(
                latents=["latent1.pt", "latent2.pt"],
                frame_start=0
            )


if __name__ == '__main__':
    unittest.main()
