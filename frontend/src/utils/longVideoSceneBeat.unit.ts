/**
 * Unit checks for scene beat parse/compose helpers.
 * Run: npx tsx frontend/src/utils/longVideoSceneBeat.unit.ts
 */
import {
  composeSceneBeat,
  parseCharacterLookBody,
  parseSceneBeat,
  splitSynopsisMood,
} from './longVideoSceneBeat';

function assertEqual<T>(actual: T, expected: T, msg: string) {
  if (JSON.stringify(actual) !== JSON.stringify(expected)) {
    throw new Error(`${msg}: expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`);
  }
}

assertEqual(
  parseSceneBeat('【中景】卧室/夜，赵今麦在卧室刷手机'),
  { shotSize: '中景', location: '卧室/夜', visual: '赵今麦在卧室刷手机' },
  'parseSceneBeat',
);

const composed = composeSceneBeat('远景', '山巅/晨', '她独自走上云雾山巅');
assertEqual(
  parseSceneBeat(composed),
  { shotSize: '远景', location: '山巅/晨', visual: '她独自走上云雾山巅' },
  'compose round-trip',
);

assertEqual(
  parseCharacterLookBody('定位：主角 | 外貌：黑色短发，白T恤'),
  { role: '主角', appearance: '黑色短发，白T恤', wardrobe: '' },
  'parseCharacterLookBody',
);

assertEqual(
  parseCharacterLookBody('定位：lead | 外貌：年轻女性 | 服装：白 T 恤 黑短裤'),
  { role: 'lead', appearance: '年轻女性', wardrobe: '白 T 恤 黑短裤' },
  'parseCharacterLookBody wardrobe',
);

assertEqual(
  splitSynopsisMood('赵今麦挑战孙悟空。\n热血逆袭，由挫败转向觉醒。'),
  { logline: '赵今麦挑战孙悟空。', mood: '热血逆袭，由挫败转向觉醒。' },
  'splitSynopsisMood',
);

console.log('longVideoSceneBeat.unit.ts: ok');
