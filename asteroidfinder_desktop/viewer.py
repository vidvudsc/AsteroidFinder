from __future__ import annotations

from pathlib import Path

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
        self.inverted = False
        self.percentile_low = 0.5
        self.percentile_high = 99.5
        self._zoom = 1.0

    @property
    def current_path(self) -> Path | None:
        return self._current_path

    def load_path(self, path: str | Path, *, keep_view: bool = False) -> None:
        image = load_image(path)
        self._current_path = image.path
        self._current_data = image.data
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
        if self._scene.sceneRect().isValid() and not self._scene.sceneRect().isEmpty():
            self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            self._zoom = 1.0

    def show_track_overlay(self, points: list[tuple[float, float]], label: str, color: str = "#38bdf8") -> None:
        if not points:
            return
        pen = QPen(QColor(color), 2.0)
        brush = QBrush(QColor(color))
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
                text = QGraphicsTextItem(label)
                text.setDefaultTextColor(QColor(color))
                text.setPos(x + 7, y + 7)
                text.setZValue(11)
                self._scene.addItem(text)
                self._overlay_items.append(text)
        for x, y in points:
            dot = self._scene.addEllipse(x - 1.8, y - 1.8, 3.6, 3.6, QPen(QColor(color)), brush)
            dot.setZValue(12)
            self._overlay_items.append(dot)

    def wheelEvent(self, event: QWheelEvent) -> None:
        factor = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
        next_zoom = self._zoom * factor
        if next_zoom < 0.05 or next_zoom > 80:
            return
        self._zoom = next_zoom
        self.scale(factor, factor)

    def _render_current(self, *, keep_view: bool) -> None:
        if self._current_data is None:
            return
        stretched = stretch_to_uint8(self._current_data, percentile=(self.percentile_low, self.percentile_high))
        if self.inverted:
            stretched = 255 - stretched
        image = _gray_qimage(stretched)
        pixmap = QPixmap.fromImage(image)
        old_transform = self.transform()
        old_center = self.mapToScene(self.viewport().rect().center())
        if self._pixmap_item is None:
            self._pixmap_item = self._scene.addPixmap(pixmap)
        else:
            self._pixmap_item.setPixmap(pixmap)
        self._scene.setSceneRect(QRectF(0, 0, pixmap.width(), pixmap.height()))
        if keep_view:
            self.setTransform(old_transform)
            self.centerOn(old_center)
        else:
            self.fit_to_view()


def _gray_qimage(data: np.ndarray) -> QImage:
    contiguous = np.ascontiguousarray(data)
    height, width = contiguous.shape
    image = QImage(contiguous.data, width, height, width, QImage.Format.Format_Grayscale8)
    return image.copy()
