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

### Theme CSS (same pattern as mac)

Import token palettes in `frontend/src/main.ts`; switch with `applyTheme()` on `<html>` (`frontend/src/utils/i18n.ts`).

| Theme | Token file (`@danqing/dq-tokens`) | `<html>` classes |
|-------|-----------------------------------|------------------|
| Apple Dark (default) | `dq-mac.css` + `dq-glass.css` | `dark` |
| Linear Dark | `dq-linear-dark.css` | `dark dq-linear-dark` |
| China Red Dark | `dq-china-red-dark.css` | `dark dq-china-red-dark` |
| Zinc Dark (shadcn-style) | `dq-shadcn-dark.css` | `dark dq-shadcn-dark` |

Studio-only chrome (sidebars, Studio 创作页浮动条, gallery) stays in `frontend/src/styles/theme-apple-*.css` — not in tokens.

`make check-consistency` runs `check_frontend_governance.py` (EP boundary, theme legacy, dq-ui compat).

## Local dev

```bash
cd ../dq-ui && pnpm install
cd frontend && npm install && npm run dev
```

Restart Vite after `dq-ui` changes.
