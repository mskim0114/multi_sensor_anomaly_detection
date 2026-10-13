"""pyzed adapter. Only this module imports the ZED SDK (ZED venv: numpy 2, pyzed, Pillow)."""
from __future__ import annotations

import time

import numpy as np


class CameraError(RuntimeError):
    pass


class ZedCamera:
    def __init__(self, config: dict):
        import pyzed.sl as sl
        self.sl = sl
        self.config = config
        self.camera = sl.Camera()
        self.runtime = sl.RuntimeParameters()
        self.image, self.depth, self.sensors = sl.Mat(), sl.Mat(), sl.SensorsData()
        width, height = config["depth_size"]
        self.depth_resolution = sl.Resolution(int(width), int(height))
        self.opened = False

    def open(self) -> dict:
        sl = self.sl
        init = sl.InitParameters(camera_resolution=getattr(sl.RESOLUTION, self.config["resolution"]),
                                 camera_fps=int(self.config["fps"]),
                                 depth_mode=getattr(sl.DEPTH_MODE, self.config["depth_mode"]),
                                 coordinate_units=sl.UNIT.METER, sdk_verbose=0)
        status = self.camera.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            raise CameraError(f"open failed: {status}")
        self.opened = True
        info = self.camera.get_camera_information()
        configuration = info.camera_configuration
        calibration = configuration.calibration_parameters

        def intrinsics(cam):
            return {"fx": cam.fx, "fy": cam.fy, "cx": cam.cx, "cy": cam.cy, "disto": list(cam.disto)}
        return {"model": str(info.camera_model), "serial_number": int(info.serial_number),
                "camera_firmware": int(configuration.firmware_version),
                "sensors_firmware": int(info.sensors_configuration.firmware_version),
                "resolution": [int(configuration.resolution.width), int(configuration.resolution.height)],
                "fps": float(configuration.fps), "depth_mode": self.config["depth_mode"],
                "baseline_mm": float(calibration.get_camera_baseline()) * 1000.0,
                "left": intrinsics(calibration.left_cam), "right": intrinsics(calibration.right_cam),
                "sdk_version": str(sl.Camera.get_sdk_version()),
                "calibration_note": "left/right intrinsics are for the capture resolution; depth is resized "
                                    "to depth_size, so scale fx/cx by width ratio and fy/cy by height ratio"}

    def grab(self) -> tuple[int, int] | None:
        """(frame monotonic ns, image timestamp ns) of the next frame, or None on failure."""
        if self.camera.grab(self.runtime) != self.sl.ERROR_CODE.SUCCESS:
            return None
        image_ns = int(self.camera.get_timestamp(self.sl.TIME_REFERENCE.IMAGE).get_nanoseconds())
        # The SDK stamps frames on the realtime clock; the collector ticks on CLOCK_MONOTONIC.
        return image_ns - (time.time_ns() - time.monotonic_ns()), image_ns

    def retrieve(self):
        sl = self.sl
        self.camera.retrieve_image(self.image, sl.VIEW.LEFT)
        self.camera.retrieve_measure(self.depth, sl.MEASURE.DEPTH, sl.MEM.CPU, self.depth_resolution)
        left = self.image.get_data()[:, :, 2::-1]          # BGRA -> RGB view
        depth = np.array(self.depth.get_data(), dtype=np.float32, copy=True)
        imu = {}
        if self.camera.get_sensors_data(self.sensors, sl.TIME_REFERENCE.IMAGE) == sl.ERROR_CODE.SUCCESS:
            data = self.sensors.get_imu_data()
            ax, ay, az = data.get_linear_acceleration()
            gx, gy, gz = data.get_angular_velocity()
            qx, qy, qz, qw = data.get_pose().get_orientation().get()
            imu = {"imu_ax": round(ax, 4), "imu_ay": round(ay, 4), "imu_az": round(az, 4),
                   "imu_gx": round(gx, 4), "imu_gy": round(gy, 4), "imu_gz": round(gz, 4),
                   "imu_qx": round(qx, 5), "imu_qy": round(qy, 5), "imu_qz": round(qz, 5), "imu_qw": round(qw, 5)}
            try:
                imu["imu_temperature_c"] = round(float(
                    self.sensors.get_temperature_data().get(sl.SENSOR_LOCATION.IMU)), 2)
            except Exception:
                pass
        return left, depth, imu

    def close(self) -> None:
        if self.opened:
            self.camera.close()
            self.opened = False
