import unittest
from types import SimpleNamespace
from unittest import mock

from src.utils.tensorflow_runtime import configure_tensorflow_runtime


class TensorFlowRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.cpu = SimpleNamespace(name="/physical_device:CPU:0")
        self.gpu = SimpleNamespace(name="/physical_device:GPU:0")

    def _list_devices(self, device_type):
        return [self.cpu] if device_type == "CPU" else []

    @mock.patch("src.utils.tensorflow_runtime.tf.config.set_visible_devices")
    @mock.patch("src.utils.tensorflow_runtime.tf.config.list_physical_devices")
    def test_cpu_is_forced_without_requiring_a_real_gpu(self, list_devices, set_visible):
        list_devices.side_effect = self._list_devices

        runtime = configure_tensorflow_runtime("cpu")

        self.assertEqual(runtime.selected_device, "cpu")
        set_visible.assert_called_once_with([], "GPU")

    @mock.patch("src.utils.tensorflow_runtime.tf.config.list_physical_devices")
    def test_gpu_mode_fails_clearly_when_no_gpu_exists(self, list_devices):
        list_devices.side_effect = self._list_devices

        with self.assertRaisesRegex(RuntimeError, "no detectó ninguna"):
            configure_tensorflow_runtime("gpu")

    @mock.patch(
        "src.utils.tensorflow_runtime.tf.config.experimental.set_memory_growth"
    )
    @mock.patch("src.utils.tensorflow_runtime.tf.config.list_physical_devices")
    def test_auto_enables_memory_growth_when_gpu_exists(self, list_devices, set_growth):
        def devices(device_type):
            return [self.cpu] if device_type == "CPU" else [self.gpu]

        list_devices.side_effect = devices

        runtime = configure_tensorflow_runtime("auto")

        self.assertEqual(runtime.selected_device, "gpu")
        self.assertEqual(runtime.memory_growth_devices, (self.gpu.name,))
        set_growth.assert_called_once_with(self.gpu, True)


if __name__ == "__main__":
    unittest.main()
