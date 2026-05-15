import { createApp } from 'vue';
import { createPinia } from 'pinia';
import ElementPlus, { ElNotification } from 'element-plus';
import * as ElementPlusIconsVue from '@element-plus/icons-vue';
import router from './router';
import i18n, { $tt, $mn, $md, $mvn, $pn, applyTheme } from './utils/i18n';
import App from './App.vue';
import './styles/theme.css';

const app = createApp(App);

// Register all Element Plus icons
for (const [key, component] of Object.entries(ElementPlusIconsVue)) {
  app.component(key, component);
}

// Global properties
app.config.globalProperties.$tt = $tt;
app.config.globalProperties.$mn = $mn;
app.config.globalProperties.$md = $md;
app.config.globalProperties.$mvn = $mvn;
app.config.globalProperties.$pn = $pn;

app.use(createPinia());
app.use(router);
app.use(ElementPlus);
app.use(i18n);

app.mount('#app');

// Global error handlers
app.config.errorHandler = (err) => {
  console.error('Vue error:', err);
  ElNotification({
    title: 'Error',
    message: String((err as Error)?.message || err).substring(0, 200),
    type: 'error',
    duration: 6000,
    position: 'bottom-right',
  });
};

window.onerror = function (msg, _source, _line, _col, error) {
  const message = (error && error.message) || String(msg);
  if (message.indexOf('ResizeObserver loop') !== -1) return true;
  ElNotification({
    title: 'JS Error',
    message: String(message).substring(0, 200),
    type: 'error',
    duration: 6000,
    position: 'bottom-right',
  });
  return false;
};

window.addEventListener('unhandledrejection', function (event) {
  const reason =
    (event.reason && event.reason.message) || event.reason || 'Unknown Promise error';
  ElNotification({
    title: 'Unhandled Promise',
    message: String(reason).substring(0, 200),
    type: 'error',
    duration: 6000,
    position: 'bottom-right',
  });
});

// Expose helpers on window for legacy compatibility
(window as unknown as Record<string, unknown>).$tt = $tt;
(window as unknown as Record<string, unknown>).$mn = $mn;
(window as unknown as Record<string, unknown>).$md = $md;
(window as unknown as Record<string, unknown>).$mvn = $mvn;
(window as unknown as Record<string, unknown>).$pn = $pn;
(window as unknown as Record<string, unknown>).DQApplyTheme = applyTheme;