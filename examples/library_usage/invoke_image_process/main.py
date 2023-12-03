import dgenerate.image_process
import shlex

args = shlex.split(
    '../../media/earth.jpg -o earth-upscaled.png --processors upscaler;model=https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth')


# the command runs inside the current process
return_code = dgenerate.image_process.invoke_image_process(args=args)


print('Return Code:', return_code)

# run the command again, this time observe render loop events

for event in dgenerate.image_process.invoke_image_process_events(args=args):
    print(event)