from typing import Any, Callable, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from pytorch3d.renderer import (
    DirectionalLights,
    FoVOrthographicCameras,
    FoVPerspectiveCameras,
    HardPhongShader,
    MeshRasterizer,
    RasterizationSettings,
    TexturesVertex,
    look_at_view_transform,
)
from pytorch3d.renderer.mesh.rasterizer import Fragments
from pytorch3d.structures import Meshes
from torch import Tensor


class FLAMEOrthographicRenderer(nn.Module):
    def __init__(self, resolution: Union[Sequence[int], int] = 224) -> None:
        super().__init__()
        self.device = torch.device("cpu")
        if isinstance(resolution, int):
            resolution = (resolution, resolution)
        self.resolution = resolution
        R, T = look_at_view_transform(2.7, 0, 0)
        cameras = FoVOrthographicCameras(R=R, T=T)
        raster_settings = RasterizationSettings(
            image_size=self.resolution,
            blur_radius=0.0,
            faces_per_pixel=1,
            perspective_correct=False,
        )
        lights = DirectionalLights(direction=[[0.0, 0.0, 1.0]])

        self.rasterizer = MeshRasterizer(
            cameras=cameras, raster_settings=raster_settings
        )
        self.shader = HardPhongShader(cameras=cameras, lights=lights)

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
        resolution: Optional[Union[Sequence[int], int]] = None,
    ) -> Tuple[Tensor, Fragments]:
        """
        Args:
            verts: shape (N, V, 3)
            faces: shape (N, F, 3)
            xy_translation: shape (N, 2) translation in the x and y directions
            scale: shape (N,) scale to apply to mesh.
            textures: Optional object of type Textures.
            resolution: Optional resolution to render at.
        Returns:
            images: shape (N, 4, H, W) RGBA images
            fragments: Fragments object
                pix_to_face: shape (N, H, W, k), k is faces_per_pixel
                zbuf: shape (N, H, W, k)
                bary_coords: shape (N, H, W, k, 3)
                dists: shape (N, H, W, k)
        """
        N = verts.size(0)
        if xy_translation is not None:
            verts[..., :2] = verts[..., :2] + xy_translation
        if scale is not None:
            verts = verts * scale
        if faces.ndim == 2 or (faces.size(0) == 1 and N > 1):
            faces = faces.expand(N, -1, -1)
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
        images: Tensor = self.shader(fragments, mesh)
        return images.permute(0, 3, 1, 2), fragments


class FLAMEPerspectiveRenderer(nn.Module):
    def __init__(self, resolution: Union[Sequence[int], int] = 224) -> None:
        super().__init__()
        self.device = torch.device("cpu")
        if isinstance(resolution, int):
            resolution = (resolution, resolution)
        self.resolution = resolution
        R, T = look_at_view_transform(2.7, 0, 0)
        cameras = FoVPerspectiveCameras(fov=6, R=R, T=T)
        raster_settings = RasterizationSettings(
            image_size=self.resolution,
            blur_radius=0.0,
            faces_per_pixel=1,
            perspective_correct=False,
        )
        lights = DirectionalLights(direction=[[0.0, 0.0, 1.0]])

        self.rasterizer = MeshRasterizer(
            cameras=cameras, raster_settings=raster_settings
        )
        self.shader = HardPhongShader(cameras=cameras, lights=lights)

    def _apply(
        self, fn: Callable[..., Any], recurse: bool = True
    ) -> "FLAMEPerspectiveRenderer":
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
        textures: Optional[Any] = None,
        resolution: Optional[Union[Sequence[int], int]] = None,
    ) -> Tuple[Tensor, Fragments]:
        """
        Args:
            verts: shape (N, V, 3)
            faces: shape (N, F, 3)
            textures: Optional object of type Textures.
            resolution: Optional resolution to render at.
        Returns:
            images: shape (N, 4, H, W) RGBA images
            fragments: Fragments object
                pix_to_face: shape (N, H, W, k), k is faces_per_pixel
                zbuf: shape (N, H, W, k)
                bary_coords: shape (N, H, W, k, 3)
                dists: shape (N, H, W, k)
        """
        N = verts.size(0)
        if faces.ndim == 2 or (faces.size(0) == 1 and N > 1):
            faces = faces.expand(N, -1, -1)
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
        images: Tensor = self.shader(fragments, mesh)
        return images.permute(0, 3, 1, 2), fragments
