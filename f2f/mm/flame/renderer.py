from typing import Any, Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
from pytorch3d.renderer import (
    DirectionalLights,
    FoVOrthographicCameras,
    MeshRasterizer,
    RasterizationSettings,
    SoftPhongShader,
    TexturesVertex,
)
from pytorch3d.structures import Meshes
from torch import Tensor


class FLAMEOrthographicRenderer(nn.Module):
    def __init__(self, resolution: int = 224) -> None:
        super().__init__()
        self.device = torch.device("cpu")
        self.resolution = resolution
        cameras = FoVOrthographicCameras(
            R=torch.eye(3, dtype=torch.float32)[None],
            T=torch.tensor([[0, 0, 3]], dtype=torch.float32),
        )
        raster_settings = RasterizationSettings(
            image_size=self.resolution,
            blur_radius=0.0,
            faces_per_pixel=1,
            perspective_correct=False,
        )
        lights = DirectionalLights(direction=[[0.0, 0.0, -1.0]])

        self.rasterizer = MeshRasterizer(
            cameras=cameras, raster_settings=raster_settings
        )
        self.shader = SoftPhongShader(cameras=cameras, lights=lights)

    def _apply(
        self, fn: Callable[..., Any], recurse: bool = True
    ) -> "FLAMEOrthographicRenderer":
        if "t" in fn.__code__.co_varnames:
            with torch.no_grad():
                null_tensor = torch.empty(0)
                device = getattr(fn(null_tensor), "device", "cpu")
            device = torch.device(device)
            if device != self.device:
                self.device = device
                self.shader.cameras.to(device)
                self.shader.lights.to(device)
                self.shader.materials.to(device)
        return super()._apply(fn)

    def forward(
        self,
        verts: Tensor,
        faces: Tensor,
        xy_translation: Optional[Tensor] = None,
        scale: Optional[Union[float, Tensor]] = None,
        textures: Optional[Any] = None,
        resolution: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:
        if xy_translation is not None:
            verts[..., :2] = verts[..., :2] + xy_translation
        if scale is not None:
            verts = verts * scale
        verts[..., (0, 2)] = -verts[..., (0, 2)]
        if faces.ndim == 2:
            faces = faces.unsqueeze(0)
        if textures is None:
            textures = TexturesVertex(torch.ones_like(verts))
        elif isinstance(textures, Tensor):
            textures = TexturesVertex(textures)
        mesh = Meshes(
            verts=verts,
            faces=faces,
            textures=textures,
        )
        if resolution is None:
            resolution = self.resolution
        self.rasterizer.raster_settings.image_size = resolution
        fragments = self.rasterizer(mesh)
        images = self.shader(fragments, mesh)
        return images, fragments
