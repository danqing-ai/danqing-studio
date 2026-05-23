# DanQing UI in Studio

Shared packages live in the sibling repo [danqing-ai/dq-ui](https://github.com/danqing-ai/dq-ui) (`file:../../dq-ui/packages/*` in `package.json`).

GitHub Actions checks out `dq-ui` next to the repo so `npm install` resolves those paths (see `.github/workflows/release.yml`).

## Feedback

```ts
import { toast, confirm } from '@/utils/feedback';

toast.success('Saved');
toast.error('Failed');
await confirm('Delete this item?', 'Confirm', { type: 'warning' });
```

- Global errors: `toast.notify({ title, message })` in `main.ts`
- Loading overlay: `v-dq-loading="isLoading"`

Hosts mount via `installDanQingFeedback` in `plugins/dq-ui.ts`.

## UI stack

| Layer | Source |
|-------|--------|
| Tokens | `@danqing/dq-tokens` (`--dq-*`) |
| Components | `@danqing/dq-ui` (`Dq*`, Reka UI) |
| Shell | `@danqing/dq-shell` |
| Icons | Lucide via `registerDqIcons` + `DqIcon` |
| Layout CSS | `studio-*` / `settings-*` classes in `theme*.css` |

`make check-consistency` runs `check-ep-boundary`, `check-theme-legacy`, and `check-ui-compat`.

## Local dev

```bash
cd ../dq-ui && pnpm install
cd frontend && npm install && npm run dev
```

Restart Vite after `dq-ui` changes.
