import pytest
import torch

from denoise_lab.tiling import reflected_pad, reflected_bayer_pad, tiled_inference

SITES=((0,0),(0,1),(1,1),(1,0))


@pytest.mark.parametrize("height,width,padding",[(1,1,(9,7,6,5)),(2,3,(4,7,2,5)),(13,19,(3,8,7,4))])
def test_bayer_padding_matches_independent_sensor_mosaic_reference(height,width,padding):
    image=torch.arange(2*4*height*width,dtype=torch.float32).reshape(2,4,height,width)
    sensor=torch.empty((2,1,2*height,2*width))
    for c,(y,x) in enumerate(SITES):sensor[:,0,y::2,x::2]=image[:,c]
    left,right,top,bottom=padding
    mirrored=reflected_pad(sensor,2*left,2*right,2*top,2*bottom)
    expected=torch.stack([mirrored[:,0,y::2,x::2] for y,x in SITES],1)
    torch.testing.assert_close(reflected_bayer_pad(image,*padding),expected,rtol=0,atol=0)


def test_packed_plane_reflection_is_not_sensor_reflection():
    image=torch.arange(4*3*4,dtype=torch.float32).reshape(1,4,3,4)
    site=reflected_bayer_pad(image,1,1,1,1)
    legacy=reflected_pad(image,1,1,1,1)
    assert not torch.equal(site,legacy)
    # The left sensor boundary reflects G1 at sensor -1 to sensor +1,
    # the first G1 site. The R site at sensor -2 maps to sensor +2 (second R).
    assert site[0,1,1,0]==image[0,1,0,0]
    assert site[0,0,1,0]==image[0,0,0,1]


def test_sensor_tiler_preserves_identity_and_rejects_unknown_map_geometry():
    image=torch.rand(1,4,19,31)
    result=tiled_inference(lambda x:x,image,tile_size=32,overlap=8,bayer_reflection=True)
    torch.testing.assert_close(result,image,rtol=1e-6,atol=1e-7)
    with pytest.raises(ValueError,match="unconditioned"):
        tiled_inference(lambda x:x,image,tile_size=32,overlap=8,bayer_reflection=True,
                        noise_map=torch.ones(1,1,19,31))


def test_default_routes_bayer_rgb_and_conditioned_maps_explicitly():
    bayer=torch.arange(4*7*9,dtype=torch.float32).reshape(1,4,7,9)
    captured=[]
    def record(tile,noise=None):
        captured.append((tile.clone(),None if noise is None else noise.clone()))
        return tile
    tiled_inference(record,bayer,tile_size=16,overlap=3)
    torch.testing.assert_close(captured.pop()[0],reflected_bayer_pad(bayer,3,4,3,6),rtol=0,atol=0)
    rgb=bayer[:,:3]
    tiled_inference(record,rgb,tile_size=16,overlap=3)
    torch.testing.assert_close(captured.pop()[0],reflected_pad(rgb,3,4,3,6),rtol=0,atol=0)
    noise=torch.arange(7*9,dtype=torch.float32).reshape(1,1,7,9)
    tiled_inference(record,bayer,tile_size=16,overlap=3,noise_map=noise)
    tile,condition=captured.pop()
    torch.testing.assert_close(tile,reflected_pad(bayer,3,4,3,6),rtol=0,atol=0)
    torch.testing.assert_close(condition,reflected_pad(noise,3,4,3,6),rtol=0,atol=0)
