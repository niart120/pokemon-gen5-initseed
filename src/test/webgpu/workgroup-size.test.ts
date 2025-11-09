import { describe, it } from 'vitest';
import { createWebGpuDeviceContext, isWebGpuSupported } from '@/lib/webgpu/seed-search/device-context';

const hasWebGpu = isWebGpuSupported();
const describeWebGpu = hasWebGpu ? describe : describe.skip;

describeWebGpu('webgpu device workgroup limits', () => {
  it('logs supported workgroup size for this device', async () => {
    const context = await createWebGpuDeviceContext({ label: 'workgroup-size-check' });
  const sizes = [undefined, 32, 64, 128, 256, 512, 1024];
    for (const size of sizes) {
      // undefined -> default 128
      const supported = context.getSupportedWorkgroupSize(size as number | undefined);
      console.info('[workgroup-limit]', {
        requested: size ?? 'default',
        supported,
      });
    }
  }, 30_000);
});
