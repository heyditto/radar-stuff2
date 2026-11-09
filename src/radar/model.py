"""Small U-Net that inpaints masked reflectivity."""

from __future__ import annotations

import torch
from torch import nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


def _residual_from_neighbors(
    raw: torch.Tensor,
    below: torch.Tensor,
    below_flag: torch.Tensor,
    above: torch.Tensor,
    above_flag: torch.Tensor,
) -> torch.Tensor:
    """Neighbor average plus a correction, or a direct prediction if neither side exists."""
    correction = torch.tanh(raw[:, 0:1])
    fallback = torch.sigmoid(raw[:, 1:2])
    count = below_flag + above_flag
    base = (below * below_flag + above * above_flag) / count.clamp(min=1.0)
    corrected = (base + correction).clamp(0.0, 1.0)
    has_neighbor = (count > 0).to(dtype=corrected.dtype)
    return has_neighbor * corrected + (1.0 - has_neighbor) * fallback


# Channel order for the gap network. The neighbor average reads only the
# reflectivity tilts. Temperature, humidity, and wind are extra inputs to
# the correction.
BELOW_CHANNEL = 2
BELOW_FLAG_CHANNEL = 3
ABOVE_CHANNEL = 4
ABOVE_FLAG_CHANNEL = 5
TEMPERATURE_CHANNEL = 6
HUMIDITY_CHANNEL = 7
WIND_U_CHANNEL = 8
WIND_V_CHANNEL = 9
ENVIRONMENT_FLAG_CHANNEL = 10
TEMPORAL_START_CHANNEL = 11
TEMPORAL_SLOT_COUNT = 12
BUOY_START_CHANNEL = TEMPORAL_START_CHANNEL + TEMPORAL_SLOT_COUNT * 2
BUOY_STATION_COUNT = 3
BUOY_FIELD_COUNT = 6
SATELLITE_CHANNEL = BUOY_START_CHANNEL + BUOY_STATION_COUNT * BUOY_FIELD_COUNT * 2
SATELLITE_FLAG_CHANNEL = SATELLITE_CHANNEL + 1
INPUT_CHANNELS = SATELLITE_FLAG_CHANNEL + 1
# The recency experiment uses only two time scans. It reuses the first two
# otherwise-empty time-slot channels for scan ages, keeping the same 73-channel
# U-Net architecture and initialization as the no-time baseline.
RECENCY_AGE_START_CHANNEL = TEMPORAL_START_CHANNEL + 4
RECENCY_INPUT_CHANNELS = INPUT_CHANNELS


def temporal_dbz_channel(slot: int) -> int:
    if not 0 <= slot < TEMPORAL_SLOT_COUNT:
        raise IndexError(slot)
    return TEMPORAL_START_CHANNEL + slot * 2


def temporal_flag_channel(slot: int) -> int:
    return temporal_dbz_channel(slot) + 1


def buoy_value_channel(station: int, field: int) -> int:
    if not 0 <= station < BUOY_STATION_COUNT:
        raise IndexError(station)
    if not 0 <= field < BUOY_FIELD_COUNT:
        raise IndexError(field)
    offset = station * BUOY_FIELD_COUNT + field
    return BUOY_START_CHANNEL + offset * 2


def buoy_flag_channel(station: int, field: int) -> int:
    return buoy_value_channel(station, field) + 1


class UNet(nn.Module):
    """Fill holes in one sweep, using the tilts below and above when they exist.

    Seventy-three channels. The first two are the visible part of the target sweep
    and the hole mask. Channels 2 through 5 are the tilt below, its flag, the
    tilt above, and its flag. A missing side is flag 0. Channels 6 through 10
    are temperature, relative humidity, eastward wind, northward wind, and a
    flag that is 1 where the sounding covers that beam height. The next 24
    channels are reflectivity/validity pairs at fixed offsets from -30 to +30
    minutes, excluding zero. The last 36 channels are six value/validity pairs
    for each of three NDBC buoys. The last two are GOES Band 13 brightness
    temperature and its quality flag. Where a
    neighbor was sampled, the output starts as the neighbor average plus a
    correction. Where neither neighbor was sampled, the output is a direct
    prediction. Spatial size must be divisible by 8. The correction head
    starts at zero.
    """

    def __init__(self, in_channels: int = INPUT_CHANNELS, base: int = 16) -> None:
        super().__init__()
        if in_channels != INPUT_CHANNELS:
            raise ValueError(f"the gap network expects {INPUT_CHANNELS} channels")
        self.enc1 = DoubleConv(in_channels, base)
        self.enc2 = DoubleConv(base, base * 2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(base * 4, base * 8)
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec3 = DoubleConv(base * 8 + base * 4, base * 4)
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = DoubleConv(base * 4 + base * 2, base * 2)
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = DoubleConv(base * 2 + base, base)
        self.head = nn.Conv2d(base, 2, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip1 = self.enc1(x)
        skip2 = self.enc2(self.pool(skip1))
        skip3 = self.enc3(self.pool(skip2))
        bottleneck = self.bottleneck(self.pool(skip3))
        decoded3 = self.dec3(torch.cat([self.up3(bottleneck), skip3], dim=1))
        decoded2 = self.dec2(torch.cat([self.up2(decoded3), skip2], dim=1))
        decoded1 = self.dec1(torch.cat([self.up1(decoded2), skip1], dim=1))
        return _residual_from_neighbors(
            self.head(decoded1),
            x[:, BELOW_CHANNEL : BELOW_CHANNEL + 1],
            x[:, BELOW_FLAG_CHANNEL : BELOW_FLAG_CHANNEL + 1],
            x[:, ABOVE_CHANNEL : ABOVE_CHANNEL + 1],
            x[:, ABOVE_FLAG_CHANNEL : ABOVE_FLAG_CHANNEL + 1],
        )
