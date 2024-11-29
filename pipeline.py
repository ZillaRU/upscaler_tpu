import os
import numpy as np
from tpu_perf.infer import SGInfer

class EngineOV:
    
    def __init__(self, model_path="", batch=1, device_id=0) :
        # 如果环境变量中没有设置device_id，则使用默认值
        if "DEVICE_ID" in os.environ:
            device_id = int(os.environ["DEVICE_ID"])
            print(">>>> device_id is in os.environ. and device_id = ",device_id)
        self.model_path = model_path
        self.model = SGInfer(model_path , batch=batch, devices=[device_id])
        self.device_id = device_id
        
    def __str__(self):
        return "EngineOV: model_path={}, device_id={}".format(self.model_path,self.device_id)
        
    def __call__(self, args):
        if isinstance(args, list):
            values = args
        elif isinstance(args, dict):
            values = list(args.values())
        else:
            raise TypeError("args is not list or dict")
        task_id = self.model.put(*values)
        task_id, results, valid = self.model.get()
        return results

def calWeight(d,k):
    '''
    :param d: 融合重叠部分直径
    :param k: 融合计算权重参数
    :return:
    '''
    x = np.arange(-d/2,d/2)
    y = 1/(1+np.exp(-k*x))
    return y

 
def imgFusion2(img1,img2, overlap,left_right=True):
    '''
    图像加权融合
    :param img1:
    :param img2:
    :param overlap: 重合长度
    :param left_right: 是否是左右融合
    :return:
    '''
    # 这里先暂时考虑平行向融合
    wei = calWeight(overlap,0.05)    # k=5 这里是超参
    h1, w1, c1 = img1.shape
    h2, w2, c2 = img2.shape
    
    if left_right:  # 左右融合
        assert h1 == h2 and c1 == c2
        img_new = np.zeros((h1, w1+w2-overlap, c1))
        img_new[:,:w1,:] = img1
        wei_expand = np.tile(wei,(h1,1))  # 权重扩增
        wei_expand = np.expand_dims(wei_expand,2).repeat(3, axis=2)
        img_new[:, w1-overlap:w1, :] = (1-wei_expand)*img1[:,w1-overlap:w1, :] + wei_expand*img2[:,:overlap, :]
        img_new[:, w1:, :]=img2[:,overlap:, :]
    else:   # 上下融合
        assert w1 == w2 and c1 == c2
        img_new = np.zeros((h1+h2-overlap, w1, c1))
        img_new[:h1, :, :] = img1
        wei = np.reshape(wei,(overlap,1))
        wei_expand = np.tile(wei,(1, w1))
        wei_expand = np.expand_dims(wei_expand,2).repeat(3, axis=2)
        img_new[h1-overlap:h1, :, :] = (1-wei_expand)*img1[h1-overlap:h1,:, :]+wei_expand*img2[:overlap,:, :]
        img_new[h1:, :, :] = img2[overlap:,:, :]
    return img_new


def imgFusion(img_list, overlap, res_w, res_h):
    print(res_w, res_h)
    pre_v_img = None
    for vi in range(len(img_list)):
        h_img = np.transpose(img_list[vi][0], (1,2,0))
        for hi in range(1, len(img_list[vi])):
            new_img = np.transpose(img_list[vi][hi], (1,2,0))
            h_img = imgFusion2(h_img, new_img, (h_img.shape[1]+new_img.shape[1]-res_w) if (hi == len(img_list[vi])-1) else overlap, True)
        pre_v_img = h_img if pre_v_img is None else imgFusion2(pre_v_img, h_img, (pre_v_img.shape[0]+h_img.shape[0]-res_h) if vi == len(img_list)-1 else overlap, False)
    return np.transpose(pre_v_img, (2,0,1))


from PIL import Image
import numpy as np
import math

class UpscaleModel: 

    def __init__(self,tile_size=(196,196),padding=4,upscale_rate=2,model=None,model_size=(200,200), device_id=0):
        self.tile_size = tile_size
        self.padding = padding
        self.upscale_rate = upscale_rate
        if model is None:
            print("use default upscaler model")
            model = "./models/other/resrgan4x.bmodel"
        self.model = EngineOV(model, device_id=device_id)
        self.model_size = model_size

    def calc_tile_position(self, width, height, col, row):
        # generate mask
        tile_left = col * self.tile_size[0]
        tile_top = row * self.tile_size[1]
        tile_right = (col + 1) * self.tile_size[0] + self.padding
        tile_bottom = (row + 1) * self.tile_size[1] + self.padding
        if tile_right > height:
            tile_right = height
            tile_left = height - self.tile_size[0] - self.padding * 1
        if tile_bottom > width:
            tile_bottom = width
            tile_top = width - self.tile_size[1] - self.padding * 1
        
        return tile_top,tile_left, tile_bottom,tile_right

    def calc_upscale_tile_position(self,tile_left, tile_top, tile_right, tile_bottom):
        return int(tile_left * self.upscale_rate), int(tile_top * self.upscale_rate), int(tile_right * self.upscale_rate), int(tile_bottom * self.upscale_rate)

    def modelprocess(self,tile):
        ntile = tile.resize(self.model_size)
        # preprocess
        ntile = np.array(ntile).astype(np.float32)
        ntile = ntile / 255
        ntile = np.transpose(ntile,(2,0,1))
        ntile = ntile[np.newaxis,:,:,:]
        
        res   = self.model([ntile])[0]
        #extract padding 
        res   = res[0]
        res   = np.transpose(res,(1,2,0))
        res   = res * 255
        res[res>255] = 255
        res[res<0] = 0
        res   = res.astype(np.uint8)
        res   = Image.fromarray(res)
        res   = res.resize(self.target_tile_size)
        return res

    def extract_and_enhance_tiles(self, image, upscale_ratio=2.0):
        if image.mode != "RGB":
            image = image.convert("RGB")
        # 获取图像的宽度和高度
        width, height = image.size
        self.upscale_rate = upscale_ratio
        self.target_tile_size = (int((self.tile_size[0]+self.padding*1) * upscale_ratio),
                                 int((self.tile_size[1]+self.padding*1) * upscale_ratio))
        target_width, target_height = int(width * upscale_ratio), int(height * upscale_ratio)
        # 计算瓦片的列数和行数
        num_cols = math.ceil((width-self.padding) / self.tile_size[0])
        num_rows = math.ceil((height-self.padding) / self.tile_size[1])
        
        # 遍历每个瓦片的行和列索引
        img_tiles = []
        for row in range(num_rows):
            img_h_tiles = []
            for col in range(num_cols):
                # 计算瓦片的左上角和右下角坐标
                tile_left, tile_top, tile_right, tile_bottom = self.calc_tile_position(width, height, row, col)
                # 裁剪瓦片
                tile = image.crop((tile_left, tile_top, tile_right, tile_bottom))
                # 使用超分辨率模型放大瓦片
                upscaled_tile = self.modelprocess(tile)
                # 将放大后的瓦片粘贴到输出图像上
                # overlap
                ntile = np.array(upscaled_tile).astype(np.float32)
                ntile = np.transpose(ntile,(2,0,1))
                img_h_tiles.append(ntile)
                
            img_tiles.append(img_h_tiles)
        res = imgFusion(img_list=img_tiles, overlap=int(self.padding*upscale_ratio), res_w=target_width, res_h=target_height)
        res = Image.fromarray(np.transpose(res,(1,2,0)).astype(np.uint8))
        return res


if __name__ == "__main__":
    import sys
    model = "./models/resrgan4x.bmodel"
    upmodel = UpscaleModel(model=model, padding=20)
    img   = Image.open(sys.argv[1])
    res   = upmodel.extract_and_enhance_tiles(img, upscale_ratio=4.0)
    res.save("./temp_out.png")
