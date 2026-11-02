/* Display cached production computations; never run an LM or fit a detector. */
'use strict';
(() => {
  const data = JSON.parse(document.getElementById('visual-data').textContent);
  const byId = id => document.getElementById(id);
  const number = value => value === null ? '未定义' : value.toFixed(3);
  const ns = 'http://www.w3.org/2000/svg';
  function svgElement(tag, attrs = {}, text) {
    const node = document.createElementNS(ns, tag);
    Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, value));
    if (text !== undefined) node.textContent = text;
    return node;
  }
  function frame(svg, width, xLimits, yLimits, xLabel, yLabel) {
    svg.replaceChildren();
    const margin = {left: 57, right: 25, top: 25, bottom: 45};
    const height = 270;
    const x = value => margin.left + (value - xLimits[0]) / (xLimits[1] - xLimits[0]) * (width - margin.left - margin.right);
    const y = value => height - margin.bottom - (value - yLimits[0]) / (yLimits[1] - yLimits[0]) * (height - margin.top - margin.bottom);
    for (let i = 0; i <= 4; i++) {
      const value = yLimits[0] + (yLimits[1] - yLimits[0]) * i / 4;
      svg.append(svgElement('line', {x1: margin.left, y1: y(value), x2: width - margin.right, y2: y(value), stroke: '#e1e8f1'}));
      svg.append(svgElement('text', {x: margin.left - 8, y: y(value) + 4, 'text-anchor': 'end', fill: '#536276', 'font-size': 12}, value.toFixed(2)));
      const xv = xLimits[0] + (xLimits[1] - xLimits[0]) * i / 4;
      svg.append(svgElement('text', {x: x(xv), y: height - 23, 'text-anchor': 'middle', fill: '#536276', 'font-size': 12}, xv.toFixed(1)));
    }
    svg.append(svgElement('text', {x: margin.left, y: 15, fill: '#536276', 'font-size': 12}, yLabel));
    svg.append(svgElement('text', {x: width - margin.right, y: height - 6, 'text-anchor': 'end', fill: '#536276', 'font-size': 12}, xLabel));
    return {x, y};
  }
  function line(svg, points, color, dashed = false) {
    svg.append(svgElement('polyline', {points: points.map(pair => pair.join(',')).join(' '),
      fill: 'none', stroke: color, 'stroke-width': 2.5, ...(dashed ? {'stroke-dasharray': '7 5'} : {})}));
  }
  function dot(svg, x, y, color, hollow = false) {
    svg.append(svgElement('circle', {cx: x, cy: y, r: 5, fill: hollow ? 'white' : color, stroke: color, 'stroke-width': 2}));
  }
  function updateBorrowing() {
    const row = data.context[Number(byId('context').value)];
    byId('context-value').textContent = row.neighbor.toFixed(1);
    byId('own-score').textContent = number(row.score[2]);
    byId('own-cavity').textContent = number(row.cavity[2]);
    byId('order-status').textContent = '本例原始观测与共享后验的片段内排序一致；自身d₃始终固定。';
    const svg = byId('borrow-chart');
    const range = data.context.map(value => value.score[2]);
    const limits = [Math.min(...range, .2) - .2, Math.max(...range, .2) + .2];
    const {x, y} = frame(svg, 520, [-3, 3], limits, '邻居观测 c', '第3个位置的分数');
    line(svg, data.context.map(value => [x(value.neighbor), y(value.score[2])]), '#245bc7');
    line(svg, [[x(-3), y(.2)], [x(3), y(.2)]], '#7542ad', true);
    dot(svg, x(row.neighbor), y(row.score[2]), '#245bc7');
  }
  function updatePooling() {
    const index = Number(byId('pooling').value);
    const row = data.pooling[index];
    byId('pooling-value').textContent = row.lambda.toFixed(2);
    byId('pooled-auc').textContent = number(row.metrics.pooled.auroc);
    byId('within-auc').textContent = number(row.metrics.within_span.auroc);
    byId('between-auc').textContent = number(row.metrics.between_spans.auroc);
    const strip = byId('token-strip');
    strip.replaceChildren();
    row.scores.forEach((value, i) => {
      const token = document.createElement('div');
      token.className = 'token';
      const level = .10 + .30 * (value - data.color_limits[0]) / (data.color_limits[1] - data.color_limits[0]);
      token.style.backgroundColor = 'rgba(36,91,199,' + level.toFixed(3) + ')';
      token.textContent = (i < 4 ? 'A' : 'B') + ((i % 4) + 1);
      const score = document.createElement('span');
      score.textContent = value.toFixed(3);
      const label = document.createElement('span');
      label.textContent = '构造标签 ' + data.labels[i];
      token.append(score, label);
      strip.append(token);
    });
    byId('pooling-status').textContent = index === 100
      ? '均值替代后：整体AUC高于原始构造分数，片段内全部ties，局部AUC=0.5。'
      : 'λ<1：同片段排序保持；整体AUC仍可能随跨片段比较而变化。';
    const svg = byId('auc-chart');
    const {x, y} = frame(svg, 920, [0, 1], [.3, 1], '均值替代程度 λ（101个离散点）', 'AUC');
    line(svg, data.pooling.map(point => [x(point.lambda), y(point.metrics.pooled.auroc)]), '#245bc7');
    line(svg, data.pooling.slice(0, -1).map(point => [x(point.lambda), y(point.metrics.within_span.auroc)]), '#7542ad', true);
    dot(svg, x(1), y(data.pooling[0].metrics.within_span.auroc), '#7542ad', true);
    dot(svg, x(1), y(.5), '#7542ad');
    dot(svg, x(row.lambda), y(row.metrics.pooled.auroc), '#245bc7');
    dot(svg, x(row.lambda), y(row.metrics.within_span.auroc), '#7542ad');
    svg.append(svgElement('text', {x: 72, y: 44, fill: '#245bc7', 'font-size': 13}, '实线：整体AUC'));
    svg.append(svgElement('text', {x: 230, y: 44, fill: '#7542ad', 'font-size': 13}, '虚线：片段内AUC；λ=1单独实心点'));
    byId('pair-counts').textContent = '本构造共有16个正负对：同片段6对，跨片段10对；同片段权重为6/16。';
    byId('shared-example').textContent = '同一组构造观测调用默认shared：整体AUC从'
      + number(data.raw_metrics.pooled.auroc) + '变为' + number(data.shared_metrics.pooled.auroc)
      + '，而片段内AUC均为' + number(data.shared_metrics.within_span.auroc)
      + '。这对应“全局变动、局部保序”；与上面的均值删除是不同操作。';
  }
  byId('context').addEventListener('input', updateBorrowing);
  byId('pooling').addEventListener('input', updatePooling);
  updateBorrowing();
  updatePooling();
  window.visualGuideReady = true;
})();
