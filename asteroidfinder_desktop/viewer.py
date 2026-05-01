from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QImage, QPainter, QPen, QPixmap, QWheelEvent
from PySide6.QtWidgets import QGraphicsEllipseItem, QGraphicsPixmapItem, QGraphicsScene, QGraphicsTextItem, QGraphicsView

from asteroidfinder.io import load_image, stretch_to_uint8


class FitsViewer(QGraphicsView):
    def __init__(self) -> None:
        super().__init__()
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setBackgroundBrush(QBrush(QColor("#080d14")))
        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._overlay_items: list[object] = []
        self._current_path: Path | None = None
        self._current_data: np.ndarray | None = None
        self._current_header: Any | None = None
        self._image_rect = QRectF()
        self.inverted = False
        self.percentile_low = 0.5
        self.percentile_high = 99.5
        self._zoom = 1.0
        self._data_cache: OrderedDict[Path, np.ndarray] = OrderedDict()
        self._header_cache: OrderedDict[Path, Any | None] = OrderedDict()
        self._pixmap_cache: OrderedDict[tuple[Path, bool, float, float], QPixmap] = OrderedDict()
        self._cache_bytes = 0
        self._max_cache_bytes = 768 * 1024 * 1024

    @property
    def current_path(self) -> Path | None:
        return self._current_path

    def load_path(self, path: str | Path, *, keep_view: bool = False) -> None:
        path = Path(path)
        cached = self._data_cache.get(path)
        if cached is None:
            image = load_image(path)
            self._current_path = image.path
            self._current_data = image.data
            self._current_header = image.header
            self._remember_data(image.path, image.data, image.header)
        else:
            self._data_cache.move_to_end(path)
            self._header_cache.move_to_end(path)
            self._current_path = path
            self._current_data = cached
            self._current_header = self._header_cache.get(path)
        self._render_current(keep_view=keep_view)

    def set_inverted(self, value: bool) -> None:
        self.inverted = value
        self._render_current(keep_view=True)

    def set_percentiles(self, low: float, high: float) -> None:
        self.percentile_low = low
        self.percentile_high = high
        self._render_current(keep_view=True)

    def clear_overlays(self) -> None:
        for item in self._overlay_items:
            self._scene.removeItem(item)
        self._overlay_items.clear()

    def fit_to_view(self) -> None:
        if self._image_rect.isValid() and not self._image_rect.isEmpty():
            self.fitInView(self._image_rect, Qt.AspectRatioMode.KeepAspectRatio)
            self._zoom = 1.0

    def show_track_overlay(
        self,
        points: list[tuple[float, float]],
        label: str,
        *,
        mode: str = "circle",
        current_index: int | None = None,
        color: str = "#38bdf8",
    ) -> None:
        if not points:
            return
        pen = QPen(QColor(color), 2.0)
        brush = QBrush(QColor(color))

        if mode in {"path", "both"}:
            previous = None
            for index, (x, y) in enumerate(points):
                marker = QGraphicsEllipseItem(QRectF(x - 5, y - 5, 10, 10))
                marker.setPen(pen)
                marker.setBrush(Qt.BrushStyle.NoBrush)
                marker.setZValue(10)
                self._scene.addItem(marker)
                self._overlay_items.append(marker)
                if previous is not None:
                    line = self._scene.addLine(previous[0], previous[1], x, y, pen)
                    line.setZValue(9)
                    self._overlay_items.append(line)
                previous = (x, y)
                if index == 0:
                    self._add_track_label(label, x, y, color)
            for x, y in points:
                dot = self._scene.addEllipse(x - 1.8, y - 1.8, 3.6, 3.6, QPen(QColor(color)), brush)
                dot.setZValue(12)
                self._overlay_items.append(dot)

        if mode in {"circle", "both"}:
            point_index = current_index if current_index is not None and 0 <= current_index < len(points) else 0
            x, y = points[point_index]
            marker = QGraphicsEllipseItem(QRectF(x - 8, y - 8, 16, 16))
            marker.setPen(QPen(QColor(color), 2.4))
            marker.setBrush(Qt.BrushStyle.NoBrush)
            marker.setZValue(20)
            self._scene.addItem(marker)
            self._overlay_items.append(marker)
            self._add_track_label(label, x, y, color)
            dot = self._scene.addEllipse(x - 1.8, y - 1.8, 3.6, 3.6, QPen(QColor(color)), brush)
            dot.setZValue(21)
            self._overlay_items.append(dot)

    def show_prediction_overlay(
        self,
        x: float,
        y: float,
        label: str,
        *,
        color: str = "#fbbf24",
    ) -> None:
        pen = QPen(QColor(color), 2.8)
        pen.setStyle(Qt.PenStyle.DashLine)
        size = 32.0
        circle = QGraphicsEllipseItem(QRectF(x - size / 2, y - size / 2, size, size))
        circle.setPen(pen)
        circle.setBrush(Qt.BrushStyle.NoBrush)
        circle.setZValue(30)
        self._scene.addItem(circle)
        self._overlay_items.append(circle)
        hline = self._scene.addLine(x - size, y, x + size, y, pen)
        vline = self._scene.addLine(x, y - size, x, y + size, pen)
        hline.setZValue(31)
        vline.setZValue(31)
        self._overlay_items.extend([hline, vline])
        self._add_track_label(label, x, y, color)

    def _add_track_label(self, label: str, x: float, y: float, color: str) -> None:
        text = QGraphicsTextItem(label)
        text.setDefaultTextColor(QColor(color))
        text.setPos(x + 9, y + 9)
        text.setZValue(22)
        self._scene.addItem(text)
        self._overlay_items.append(text)

    def wheelEvent(self, event: QWheelEvent) -> None:
        factor = 1.08 if event.angleDelta().y() > 0 else 1 / 1.08
        next_zoom = self._zoom * factor
        if next_zoom < 0.35 or next_zoom > 35:
            return
        self._zoom = next_zoom
        self.scale(factor, factor)

    def _render_current(self, *, keep_view: bool) -> None:
        if self._current_data is None:
            return
        pixmap = self._current_pixmap()
        old_transform = self.transform()
        old_center = self.mapToScene(self.viewport().rect().center())
        if self._pixmap_item is None:
            self._pixmap_item = self._scene.addPixmap(pixmap)
        else:
            self._pixmap_item.setPixmap(pixmap)
        self._image_rect = QRectF(0, 0, pixmap.width(), pixmap.height())
        pad_x = max(80.0, pixmap.width() * 0.18)
        pad_y = max(80.0, pixmap.height() * 0.18)
        self._scene.setSceneRect(QRectF(-pad_x, -pad_y, pixmap.width() + pad_x * 2, pixmap.height() + pad_y * 2))
        if keep_view:
            self.setTransform(old_transform)
            self.centerOn(old_center)
        else:
            self.fit_to_view()

    def _current_pixmap(self) -> QPixmap:
        if self._current_data is None or self._current_path is None:
            return QPixmap()
        key = (self._current_path, self.inverted, self.percentile_low, self.percentile_high)
        cached = self._pixmap_cache.get(key)
        if cached is not None:
            self._pixmap_cache.move_to_end(key)
            return cached
        display_data = _display_luminance(self._current_data, self._current_header)
        stretched = stretch_to_uint8(display_data, percentile=(self.percentile_low, self.percentile_high))
        if self.inverted:
            stretched = 255 - stretched
        pixmap = QPixmap.fromImage(_gray_qimage(stretched))
        self._pixmap_cache[key] = pixmap
        self._pixmap_cache.move_to_end(key)
        self._trim_cache()
        return pixmap

    def _remember_data(self, path: Path, data: np.ndarray, header: Any | None) -> None:
        old = self._data_cache.pop(path, None)
        if old is not None:
            self._cache_bytes -= int(old.nbytes)
        self._data_cache[path] = data
        self._header_cache[path] = header
        self._data_cache.move_to_end(path)
        self._header_cache.move_to_end(path)
        self._cache_bytes += int(data.nbytes)
        self._trim_cache()

    def _trim_cache(self) -> None:
        while self._cache_bytes > self._max_cache_bytes and len(self._data_cache) > 1:
            path, data = self._data_cache.popitem(last=False)
            self._header_cache.pop(path, None)
            self._cache_bytes -= int(data.nbytes)
            for key in list(self._pixmap_cache):
                if key[0] == path:
                    self._pixmap_cache.pop(key, None)
        while len(self._pixmap_cache) > max(len(self._data_cache) * 4, 8):
            self._pixmap_cache.popitem(last=False)


def _gray_qimage(data: np.ndarray) -> QImage:
    contiguous = np.ascontiguousarray(data)
    height, width = contiguous.shape
    image = QImage(contiguous.data, width, height, width, QImage.Format.Format_Grayscale8)
    return image.copy()


def _display_luminance(data: np.ndarray, header: Any | None = None) -> np.ndarray:
    if data.ndim != 2 or not _is_bayer_mosaic(header):
        return data
    return _smooth_bayer_luminance(data)


def _is_bayer_mosaic(header: Any | None) -> bool:
    if header is None:
        return False
    bayer = str(header.get("BAYERPAT", "")).strip().upper()
    if bayer and bayer not in {"INVALID", "NONE", "FALSE", "0"}:
        return True
    color_type = str(header.get("COLORTYP", "")).strip().upper()
    return color_type in {"RGGB", "BGGR", "GRBG", "GBRG"}


def _smooth_bayer_luminance(data: np.ndarray) -> np.ndarray:
    """Build a same-size preview luminance from 2x2 CFA cells.

    Raw one-shot-color FITS frames store red/green/blue samples in a Bayer grid.
    Displaying those samples directly creates a checkerboard that aliases while
    zooming. For preview only, average each 2x2 cell and broadcast it back to
    the original pixel grid so overlay coordinates and image dimensions stay
    unchanged.
    """

    arr = np.asarray(data, dtype=np.float32)
    height, width = arr.shape
    even_height = height - height % 2
    even_width = width - width % 2
    if even_height < 2 or even_width < 2:
        return arr
    preview = arr.copy()
    cells = arr[:even_height, :even_width].reshape(even_height // 2, 2, even_width // 2, 2)
    averaged = np.nanmean(cells, axis=(1, 3))
    preview[:even_height:2, :even_width:2] = averaged
    preview[1:even_height:2, :even_width:2] = averaged
    preview[:even_height:2, 1:even_width:2] = averaged
    preview[1:even_height:2, 1:even_width:2] = averaged
    return preview
