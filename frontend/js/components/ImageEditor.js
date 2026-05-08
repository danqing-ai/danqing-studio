/**
 * 图像编辑器组件（Plan C6 蒙版画布）
 * 支持画笔、套索、橡皮擦遮罩绘制，导出黑白遮罩图；键盘快捷键绑在根节点（非 window）。
 * 用于 Inpainting（局部重绘）模式；空态仅文案引导，上传/资产库/最近由创作页卡片头部 AssetPicker 提供（Plan C4）。
 */

const ImageEditor = {
    template: `
        <div class="image-editor" ref="editorRoot" tabindex="0" @keydown="onKeyDown">
            <div class="editor-canvas-wrapper" ref="canvasWrapper">
                <canvas
                    ref="mainCanvas"
                    @mousedown="onPointerDown"
                    @mousemove="onPointerMove"
                    @mouseup="onPointerUp"
                    @mouseleave="onPointerUp"
                    @touchstart="onTouchStart"
                    @touchmove.prevent="onTouchMove"
                    @touchend="onTouchEnd"
                    @wheel.prevent="onWheel"
                    class="editor-canvas"
                ></canvas>
                <div v-if="!src" class="editor-empty">
                    <el-icon size="48"><picture-filled /></el-icon>
                    <p class="editor-empty__lead">{{ $t('studio.uploadEditImage') }}</p>
                    <p class="editor-empty__subhint">{{ $t('studio.editImageEmptyHint') }}</p>
                </div>
            </div>
            <div v-if="src" class="editor-toolbar">
                <div class="toolbar-group">
                    <el-button-group>
                        <el-button
                            :type="currentTool === 'brush' ? 'primary' : ''"
                            @click="setTool('brush')"
                            :title="$t('studio.brush')"
                            size="small"
                        >
                            <el-icon><brush /></el-icon>
                        </el-button>
                        <el-button
                            :type="currentTool === 'lasso' ? 'primary' : ''"
                            @click="setTool('lasso')"
                            :title="$t('studio.lasso')"
                            size="small"
                        >
                            <el-icon><Edit /></el-icon>
                        </el-button>
                        <el-button
                            :type="currentTool === 'eraser' ? 'primary' : ''"
                            @click="setTool('eraser')"
                            :title="$t('studio.eraser')"
                            size="small"
                        >
                            <el-icon><Delete /></el-icon>
                        </el-button>
                    </el-button-group>
                </div>
                <div class="toolbar-group" v-if="currentTool === 'brush' || currentTool === 'eraser'">
                    <span class="toolbar-label">{{ $t('studio.brushSize') }}</span>
                    <el-slider
                        v-model="brushSize"
                        :min="4"
                        :max="200"
                        :step="2"
                        style="width: 120px;"
                        size="small"
                    />
                    <span class="toolbar-val">{{ brushSize }}px</span>
                </div>
                <div class="toolbar-group" v-if="currentTool === 'lasso' && lassoPoints.length > 0">
                    <el-button size="small" @click="closeLasso">{{ $t('studio.closeLasso') }}</el-button>
                </div>
                <div class="toolbar-group" v-if="hasMaskContent">
                    <el-button size="small" @click="invertMask">{{ $t('studio.invertMask') }}</el-button>
                </div>
                <div class="toolbar-group">
                    <el-button size="small" :disabled="undoStack.length === 0" @click="undo" :title="$t('studio.undo') + ' Ctrl+Z'">
                        <el-icon><RefreshLeft /></el-icon>
                    </el-button>
                    <el-button size="small" :disabled="redoStack.length === 0" @click="redo" :title="$t('studio.redo') + ' Ctrl+Shift+Z'">
                        <el-icon><RefreshRight /></el-icon>
                    </el-button>
                </div>
                <div class="toolbar-group">
                    <el-button size="small" type="danger" text @click="clearMask" :disabled="!hasMaskContent">{{ $t('studio.clearMask') }}</el-button>
                </div>
                <div class="toolbar-divider"></div>
                <div class="toolbar-group">
                    <el-button size="small" circle @click="zoomOut" :disabled="zoom <= 0.25" title="缩小">
                        <el-icon><zoom-out /></el-icon>
                    </el-button>
                    <span class="toolbar-val" style="min-width: 48px; text-align: center;">{{ Math.round(zoom * 100) }}%</span>
                    <el-button size="small" circle @click="zoomIn" :disabled="zoom >= 4" title="放大">
                        <el-icon><zoom-in /></el-icon>
                    </el-button>
                    <el-button size="small" circle @click="resetView" title="重置视图">
                        <el-icon><refresh /></el-icon>
                    </el-button>
                </div>
            </div>
        </div>
    `,
    props: {
        src: { type: String, default: '' },
        mode: { type: String, default: 'image_to_image' }, // "image_to_image" | "inpainting"
        recentGallery: { type: Array, default: () => [] },
    },
    emits: ['pick-edit-source'],
    setup(props, { emit, expose }) {
        const { ref, reactive, onMounted, nextTick, watch } = Vue;

        const editorRoot = ref(null);
        const mainCanvas = ref(null);
        let img = null;
        let maskCtx = null;
        let mainCtx = null;
        let offscreenCanvas = null;
        let displayWidth = 0;
        let displayHeight = 0;
        let scaleX = 1;
        let scaleY = 1;

        const currentTool = ref('brush');
        const brushSize = ref(40);
        const undoStack = ref([]);
        const redoStack = ref([]);
        const hasMaskContent = ref(false);
        const lassoPoints = ref([]);
        let isDrawing = false;

        // 视图变换状态
        const zoom = ref(1.0);
        const panX = ref(0);
        const panY = ref(0);
        let isPanning = false;
        let lastPanX = 0;
        let lastPanY = 0;
        let isPinching = false;
        let lastPinchDist = 0;
        let lastPinchCenter = { x: 0, y: 0 };

        const MAX_UNDO = 20;

        function pushUndo() {
            if (!offscreenCanvas) return;
            const data = offscreenCanvas.toDataURL();
            undoStack.value.push(data);
            if (undoStack.value.length > MAX_UNDO) {
                undoStack.value.shift();
            }
            redoStack.value = [];
            updateHasMaskContent();
        }

        function updateHasMaskContent() {
            if (!offscreenCanvas) {
                hasMaskContent.value = false;
                return;
            }
            const ctx = offscreenCanvas.getContext('2d');
            const imageData = ctx.getImageData(0, 0, offscreenCanvas.width, offscreenCanvas.height);
            for (let i = 3; i < imageData.data.length; i += 4) {
                if (imageData.data[i] > 0) {
                    hasMaskContent.value = true;
                    return;
                }
            }
            hasMaskContent.value = false;
        }

        function draw() {
            if (!mainCtx || !img) return;
            const w = displayWidth;
            const h = displayHeight;
            mainCtx.clearRect(0, 0, w, h);

            mainCtx.save();
            mainCtx.translate(panX.value, panY.value);
            mainCtx.scale(zoom.value, zoom.value);

            // 绘制背景图
            mainCtx.drawImage(img, 0, 0, displayWidth, displayHeight);

            // 绘制遮罩
            if (offscreenCanvas) {
                mainCtx.save();
                mainCtx.globalAlpha = 0.5;
                mainCtx.drawImage(offscreenCanvas, 0, 0, displayWidth, displayHeight);
                mainCtx.restore();
            }

            // 绘制套索
            if (currentTool.value === 'lasso' && lassoPoints.value.length > 0) {
                mainCtx.save();
                mainCtx.strokeStyle = '#e94560';
                mainCtx.lineWidth = 2 / zoom.value;
                mainCtx.setLineDash([4 / zoom.value, 4 / zoom.value]);
                mainCtx.beginPath();
                mainCtx.moveTo(lassoPoints.value[0].x, lassoPoints.value[0].y);
                for (let i = 1; i < lassoPoints.value.length; i++) {
                    mainCtx.lineTo(lassoPoints.value[i].x, lassoPoints.value[i].y);
                }
                const last = lassoPoints.value[lassoPoints.value.length - 1];
                mainCtx.lineTo(last.x, last.y);
                mainCtx.stroke();
                mainCtx.setLineDash([]);
                mainCtx.restore();
            }

            mainCtx.restore();
        }

        function fitCanvas() {
            if (!img || !mainCanvas.value) return;
            const container = mainCanvas.value.parentElement;
            const maxW = container.clientWidth - 4;
            const maxH = Math.min(maxW, window.innerHeight * 0.6);
            let w, h;
            if (img.width / img.height > maxW / maxH) {
                w = maxW;
                h = (img.height / img.width) * maxW;
            } else {
                h = maxH;
                w = (img.width / img.height) * maxH;
            }
            displayWidth = Math.floor(w);
            displayHeight = Math.floor(h);
            scaleX = img.width / displayWidth;
            scaleY = img.height / displayHeight;
            mainCanvas.value.width = displayWidth;
            mainCanvas.value.height = displayHeight;
            mainCtx = mainCanvas.value.getContext('2d');
            offscreenCanvas = document.createElement('canvas');
            offscreenCanvas.width = displayWidth;
            offscreenCanvas.height = displayHeight;
            maskCtx = offscreenCanvas.getContext('2d');
            maskCtx.fillStyle = '#e94560';
            draw();
        }

        function loadImage(srcUrl) {
            img = new Image();
            img.onload = () => {
                fitCanvas();
            };
            img.src = srcUrl;
        }

        function getCanvasPos(e) {
            const rect = mainCanvas.value.getBoundingClientRect();
            let clientX, clientY;
            if (e.touches && e.touches.length > 0) {
                clientX = e.touches[0].clientX;
                clientY = e.touches[0].clientY;
            } else {
                clientX = e.clientX;
                clientY = e.clientY;
            }
            const x = (clientX - rect.left - panX.value) / zoom.value;
            const y = (clientY - rect.top - panY.value) / zoom.value;
            return { x, y };
        }

        function drawBrushAt(x, y) {
            if (!maskCtx) return;
            maskCtx.save();
            maskCtx.globalCompositeOperation = currentTool.value === 'eraser' ? 'destination-out' : 'source-over';
            maskCtx.fillStyle = '#e94560';
            maskCtx.beginPath();
            const radius = (brushSize.value / 2) / zoom.value;
            maskCtx.arc(x, y, radius, 0, Math.PI * 2);
            maskCtx.fill();
            maskCtx.restore();
        }

        function onPointerDown(e) {
            if (!img || !maskCtx) return;
            try {
                if (editorRoot.value && typeof editorRoot.value.focus === 'function') {
                    editorRoot.value.focus();
                }
            } catch (err) { /* ignore */ }
            // 中键拖拽平移
            if (e.button === 1 || (e.button === 0 && e.ctrlKey)) {
                isPanning = true;
                lastPanX = e.clientX;
                lastPanY = e.clientY;
                if (mainCanvas.value) mainCanvas.value.style.cursor = 'grabbing';
                e.preventDefault();
                return;
            }
            if (e.button !== 0) return;
            isDrawing = true;
            const pos = getCanvasPos(e);
            if (currentTool.value === 'brush' || currentTool.value === 'eraser') {
                pushUndo();
                drawBrushAt(pos.x, pos.y);
                draw();
            } else if (currentTool.value === 'lasso') {
                lassoPoints.value.push(pos);
                draw();
            }
        }

        function onPointerMove(e) {
            if (isPanning) {
                const dx = e.clientX - lastPanX;
                const dy = e.clientY - lastPanY;
                panX.value += dx;
                panY.value += dy;
                lastPanX = e.clientX;
                lastPanY = e.clientY;
                draw();
                e.preventDefault();
                return;
            }
            if (!isDrawing || !maskCtx) return;
            const pos = getCanvasPos(e);
            if (currentTool.value === 'brush' || currentTool.value === 'eraser') {
                drawBrushAt(pos.x, pos.y);
                draw();
            }
        }

        function onPointerUp() {
            if (isPanning) {
                isPanning = false;
                if (mainCanvas.value) mainCanvas.value.style.cursor = currentTool.value === 'brush' || currentTool.value === 'eraser' ? 'crosshair' : 'crosshair';
                return;
            }
            if (!isDrawing) return;
            isDrawing = false;
            if (currentTool.value === 'brush' || currentTool.value === 'eraser') {
                updateHasMaskContent();
            }
        }

        function onTouchStart(e) {
            if (!img || !maskCtx) return;
            try {
                if (editorRoot.value && typeof editorRoot.value.focus === 'function') {
                    editorRoot.value.focus();
                }
            } catch (err) { /* ignore */ }
            e.preventDefault();
            if (e.touches.length === 2) {
                isPinching = true;
                isDrawing = false;
                const dx = e.touches[0].clientX - e.touches[1].clientX;
                const dy = e.touches[0].clientY - e.touches[1].clientY;
                lastPinchDist = Math.sqrt(dx * dx + dy * dy);
                lastPinchCenter = {
                    x: (e.touches[0].clientX + e.touches[1].clientX) / 2,
                    y: (e.touches[0].clientY + e.touches[1].clientY) / 2
                };
                return;
            }
            isDrawing = true;
            const pos = getCanvasPos(e);
            if (currentTool.value === 'brush' || currentTool.value === 'eraser') {
                pushUndo();
                drawBrushAt(pos.x, pos.y);
                draw();
            } else if (currentTool.value === 'lasso') {
                lassoPoints.value.push(pos);
                draw();
            }
        }

        function onTouchMove(e) {
            e.preventDefault();
            if (e.touches.length === 2 && isPinching) {
                const dx = e.touches[0].clientX - e.touches[1].clientX;
                const dy = e.touches[0].clientY - e.touches[1].clientY;
                const dist = Math.sqrt(dx * dx + dy * dy);
                const center = {
                    x: (e.touches[0].clientX + e.touches[1].clientX) / 2,
                    y: (e.touches[0].clientY + e.touches[1].clientY) / 2
                };
                const scale = dist / lastPinchDist;
                const newZoom = Math.max(0.25, Math.min(4.0, zoom.value * scale));
                const rect = mainCanvas.value.getBoundingClientRect();
                const canvasCenterX = center.x - rect.left;
                const canvasCenterY = center.y - rect.top;
                const worldX = (canvasCenterX - panX.value) / zoom.value;
                const worldY = (canvasCenterY - panY.value) / zoom.value;
                panX.value = canvasCenterX - worldX * newZoom;
                panY.value = canvasCenterY - worldY * newZoom;
                zoom.value = newZoom;
                lastPinchDist = dist;
                draw();
                return;
            }
            if (!isDrawing || !maskCtx) return;
            const pos = getCanvasPos(e);
            if (currentTool.value === 'brush' || currentTool.value === 'eraser') {
                drawBrushAt(pos.x, pos.y);
                draw();
            }
        }

        function onTouchEnd(e) {
            e.preventDefault();
            if (e.touches.length < 2) {
                isPinching = false;
            }
            if (!isDrawing) return;
            isDrawing = false;
            if (currentTool.value === 'brush' || currentTool.value === 'eraser') {
                updateHasMaskContent();
            }
        }

        function onWheel(e) {
            if (!img) return;
            e.preventDefault();
            const rect = mainCanvas.value.getBoundingClientRect();
            const mouseX = e.clientX - rect.left;
            const mouseY = e.clientY - rect.top;
            const worldX = (mouseX - panX.value) / zoom.value;
            const worldY = (mouseY - panY.value) / zoom.value;
            const delta = e.deltaY > 0 ? 0.9 : 1.1;
            const newZoom = Math.max(0.25, Math.min(4.0, zoom.value * delta));
            panX.value = mouseX - worldX * newZoom;
            panY.value = mouseY - worldY * newZoom;
            zoom.value = newZoom;
            draw();
        }

        function zoomIn() {
            if (!img) return;
            const newZoom = Math.min(4.0, zoom.value * 1.25);
            const cx = displayWidth / 2;
            const cy = displayHeight / 2;
            const worldX = (cx - panX.value) / zoom.value;
            const worldY = (cy - panY.value) / zoom.value;
            panX.value = cx - worldX * newZoom;
            panY.value = cy - worldY * newZoom;
            zoom.value = newZoom;
            draw();
        }

        function zoomOut() {
            if (!img) return;
            const newZoom = Math.max(0.25, zoom.value / 1.25);
            const cx = displayWidth / 2;
            const cy = displayHeight / 2;
            const worldX = (cx - panX.value) / zoom.value;
            const worldY = (cy - panY.value) / zoom.value;
            panX.value = cx - worldX * newZoom;
            panY.value = cy - worldY * newZoom;
            zoom.value = newZoom;
            draw();
        }

        function resetView() {
            zoom.value = 1.0;
            panX.value = 0;
            panY.value = 0;
            draw();
        }

        function setTool(tool) {
            currentTool.value = tool;
            lassoPoints.value = [];
            draw();
            mainCanvas.value.style.cursor = tool === 'brush' || tool === 'eraser' ? 'crosshair' : 'crosshair';
        }

        function closeLasso() {
            if (lassoPoints.value.length < 3) return;
            pushUndo();
            const ctx = maskCtx;
            ctx.save();
            ctx.fillStyle = '#e94560';
            ctx.beginPath();
            ctx.moveTo(lassoPoints.value[0].x, lassoPoints.value[0].y);
            for (let i = 1; i < lassoPoints.value.length; i++) {
                ctx.lineTo(lassoPoints.value[i].x, lassoPoints.value[i].y);
            }
            ctx.closePath();
            ctx.fill();
            ctx.restore();
            lassoPoints.value = [];
            updateHasMaskContent();
            draw();
        }

        function invertMask() {
            if (!offscreenCanvas) return;
            pushUndo();
            const ctx = offscreenCanvas.getContext('2d');
            const imageData = ctx.getImageData(0, 0, offscreenCanvas.width, offscreenCanvas.height);
            for (let i = 3; i < imageData.data.length; i += 4) {
                imageData.data[i] = imageData.data[i] > 0 ? 0 : 255;
            }
            ctx.putImageData(imageData, 0, 0);
            updateHasMaskContent();
            draw();
        }

        function clearMask() {
            if (!offscreenCanvas) return;
            pushUndo();
            maskCtx.clearRect(0, 0, offscreenCanvas.width, offscreenCanvas.height);
            hasMaskContent.value = false;
            draw();
        }

        function undo() {
            if (undoStack.value.length === 0) return;
            redoStack.value.push(offscreenCanvas.toDataURL());
            const prev = undoStack.value.pop();
            const tempImg = new Image();
            tempImg.onload = () => {
                maskCtx.clearRect(0, 0, offscreenCanvas.width, offscreenCanvas.height);
                maskCtx.drawImage(tempImg, 0, 0);
                updateHasMaskContent();
                draw();
            };
            tempImg.src = prev;
        }

        function redo() {
            if (redoStack.value.length === 0) return;
            undoStack.value.push(offscreenCanvas.toDataURL());
            const next = redoStack.value.pop();
            const tempImg = new Image();
            tempImg.onload = () => {
                maskCtx.clearRect(0, 0, offscreenCanvas.width, offscreenCanvas.height);
                maskCtx.drawImage(tempImg, 0, 0);
                updateHasMaskContent();
                draw();
            };
            tempImg.src = next;
        }

        function onKeyDown(e) {
            if ((e.ctrlKey || e.metaKey) && e.key === 'z') {
                e.preventDefault();
                if (e.shiftKey) {
                    redo();
                } else {
                    undo();
                }
            }
            if (e.key === 'Enter' && currentTool.value === 'lasso' && lassoPoints.value.length >= 3) {
                e.preventDefault();
                closeLasso();
            }
        }

        function getMaskBlob() {
            return new Promise((resolve) => {
                if (!offscreenCanvas) {
                    resolve(null);
                    return;
                }
                offscreenCanvas.toBlob((blob) => {
                    resolve(blob);
                }, 'image/png');
            });
        }

        onMounted(() => {
            if (props.src) {
                loadImage(props.src);
            }
        });

        watch(() => props.src, (newSrc) => {
            if (newSrc) {
                undoStack.value = [];
                redoStack.value = [];
                hasMaskContent.value = false;
                lassoPoints.value = [];
                zoom.value = 1.0;
                panX.value = 0;
                panY.value = 0;
                loadImage(newSrc);
            }
        });

        expose({ getMaskBlob, clearMask });

        return {
            editorRoot,
            mainCanvas,
            onKeyDown,
            currentTool,
            brushSize,
            hasMaskContent,
            lassoPoints,
            undoStack,
            redoStack,
            setTool,
            closeLasso,
            invertMask,
            clearMask,
            undo,
            redo,
            onPointerDown,
            onPointerMove,
            onPointerUp,
            onTouchStart,
            onTouchMove,
            onTouchEnd,
            onWheel,
            zoomIn,
            zoomOut,
            resetView,
            zoom
        };
    }
};
