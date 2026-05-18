import typing
from typing import Any, Self
from nzcvm.components import Component
import dask.array as da
import xarray as xr


class Qualities(xr.Dataset):
    """
    A typed subclass of xarray.Dataset that enforces specific coordinate variables
    and guarantees that updates are strictly lazy (Dask-backed).
    """

    __slots__ = ()

    @property
    def rho(self) -> xr.DataArray:
        return self["rho"]

    @rho.setter
    def rho(self, value: xr.DataArray) -> None:
        self["rho"] = value

    @property
    def vp(self) -> xr.DataArray:
        return self["vp"]

    @vp.setter
    def vp(self, value: xr.DataArray) -> None:
        self["vp"] = value

    @property
    def vs(self) -> xr.DataArray:
        return self["vs"]

    @vs.setter
    def vs(self, value: xr.DataArray) -> None:
        self["vs"] = value

    @property
    def qp(self) -> xr.DataArray:
        return self["qp"]

    @qp.setter
    def qp(self, value: xr.DataArray) -> None:
        self["qp"] = value

    @property
    def qs(self) -> xr.DataArray:
        return self["qs"]

    @qs.setter
    def qs(self, value: xr.DataArray) -> None:
        self["qs"] = value

    @property
    def alpha(self) -> xr.DataArray:
        return self["alpha"]

    @alpha.setter
    def alpha(self, value: xr.DataArray) -> None:
        self["alpha"] = value

    @classmethod
    def from_dataset(cls, ds: xr.Dataset) -> Self:
        """Constructs a Qualities instance directly from an xarray Dataset."""

        if isinstance(ds, Qualities):
            return typing.cast(Self, ds)

        required = {"rho", "vp", "vs", "qp", "qs", "alpha"}
        missing = required - set(ds.data_vars)
        if missing:
            raise ValueError(
                f"Dataset is missing required quality variables: {missing}"
            )

        # Create a shallow copy and re-assign the class type
        obj = ds.copy(deep=False)
        obj.__class__ = cls

        return typing.cast(Self, obj)

    def __getitem__(self, key: Any) -> Any:
        # Support both qualities[Component.RHO] and native qualities["rho"]
        if isinstance(key, Component):
            key = key.name.lower()
        return super().__getitem__(key)

    def __setitem__(self, key: Any, value: Any) -> None:
        # Intercept dictionary-style assignment to validate lazy requirements
        target_key = key.name.lower() if isinstance(key, Component) else str(key)

        lazy_fields = {"rho", "vp", "vs", "qp", "qs", "alpha"}
        if target_key in lazy_fields:
            self._assert_lazy(target_key, value)

        super().__setitem__(target_key, value)

    def __setattr__(self, name: str, value: Any) -> None:
        # This catches direct dot assignments like qualities.rho = data
        lazy_fields = {"rho", "vp", "vs", "qp", "qs", "alpha"}
        if name in lazy_fields:
            self._assert_lazy(name, value)

        super().__setattr__(name, value)

    def _assert_lazy(self, name: str, value: Any) -> None:
        """Helper to ensure input payloads are wrapped with Dask."""
        underlying_data = getattr(value, "data", None)
        if not isinstance(underlying_data, da.Array):
            raise ValueError(
                f"Attribute '{name}' must be lazy (backed by a Dask Array). "
                f"Got {type(underlying_data).__name__} instead."
            )

    def blend(self, rhs: Self) -> Self:
        """
        Blends this quality layer with another layer using alpha compositing.
        Assumes self is the foreground layer and rhs is the background layer.
        """
        # Calculate the blended alpha array/value
        blended_alpha = self.alpha + rhs.alpha * (1.0 - self.alpha)

        # Calculate weighting coefficients
        a0 = self.alpha / blended_alpha
        a1 = rhs.alpha * (1.0 - self.alpha) / blended_alpha

        # Compile the new data variables into a fresh xarray Dataset
        blended_ds = xr.Dataset(
            data_vars={
                "rho": a0 * self.rho + a1 * rhs.rho,
                "vp": a0 * self.vp + a1 * rhs.vp,
                "vs": a0 * self.vs + a1 * rhs.vs,
                "qp": a0 * self.qp + a1 * rhs.qp,
                "qs": a0 * self.qs + a1 * rhs.qs,
                "alpha": blended_alpha,
            },
            # Inherit spatial tracking metadata if needed, prioritizing foreground
            attrs=self.attrs.copy(),
        )

        # Cast the plain dataset into your strict, validated Qualities class
        return self.from_dataset(blended_ds)


def template_like(arr: xr.DataArray) -> xr.Dataset:
    return xr.Dataset({component: arr for component in list(Component)})
