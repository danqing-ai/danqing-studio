/**
 * Unit checks for registry resolution helpers.
 * Run: npx tsx frontend/src/utils/registryParamSchema.unit.ts
 */
import {
  buildResolutionSizeOptions,
  pickClosestResolutionPreset,
  type ResolutionSizeOption,
} from './registryParamSchema';

function assert(cond: boolean, msg: string) {
  if (!cond) throw new Error(msg);
}

const ltxParams = {
  resolution_presets: {
    default: '512x768',
    options: [
      '480x704',
      '512x768',
      '704x1280',
      '768x1280',
      '1280x704',
      '1280x768',
    ],
  },
};

const options = buildResolutionSizeOptions(ltxParams);
assert(options.some((o) => o.value === '768x1280'), 'missing 768x1280 preset');
assert(!options.some((o) => o.value.startsWith('720x')), '720 presets must be removed');

const portraitPick = pickClosestResolutionPreset(options, 768, 1280);
assert(portraitPick === '768x1280', `768x1280 source expected 768x1280, got ${portraitPick}`);

const landscapePick = pickClosestResolutionPreset(options, 1280, 768);
assert(landscapePick === '1280x768', `1280x768 source expected 1280x768, got ${landscapePick}`);

const nearPick = pickClosestResolutionPreset(options, 720, 1280);
assert(nearPick === '704x1280', `720x1280 source expected closest 704x1280, got ${nearPick}`);

console.log('registryParamSchema.unit: ok');
