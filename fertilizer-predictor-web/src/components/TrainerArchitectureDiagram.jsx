import React from 'react';
import ReactFlow, { Background, Controls } from 'reactflow';

const nodes = [
  { id: 'input', position: { x: 0, y: 100 }, data: { label: 'Input Layer\n(Raw Data)' }, style: { width: 160 }, type: 'input' },
  { id: 'preprocessing', position: { x: 220, y: 100 }, data: { label: 'Preprocessing &\nFeature Engineering' }, style: { width: 200 } },
  { id: 'kfold', position: { x: 480, y: 100 }, data: { label: 'K-Fold Cross-Validation' }, style: { width: 180 } },
  { id: 'lgbm', position: { x: 800, y: 20 }, data: { label: 'LightGBM' }, style: { width: 100, background: '#f1f5f9' } },
  { id: 'catboost', position: { x: 800, y: 100 }, data: { label: 'CatBoost' }, style: { width: 100, background: '#f1f5f9' } },
  { id: 'xgboost', position: { x: 800, y: 180 }, data: { label: 'XGBoost' }, style: { width: 100, background: '#f1f5f9' } },
  { id: 'stacking', position: { x: 1100, y: 100 }, data: { label: 'Stacking Layer\n(Meta-Model)' }, style: { width: 170 } },
  { id: 'artifacts', position: { x: 1100, y: 200 }, data: { label: 'Artifact Saving' }, style: { width: 170 } },
];

const edges = [
  { id: 'e1', source: 'input', target: 'preprocessing', animated: true },
  { id: 'e2', source: 'preprocessing', target: 'kfold', animated: true },
  { id: 'e3', source: 'kfold', target: 'lgbm', label: 'Base Model', type: 'smoothstep' },
  { id: 'e4', source: 'kfold', target: 'catboost', label: 'Base Model', type: 'smoothstep' },
  { id: 'e5', source: 'kfold', target: 'xgboost', label: 'Base Model', type: 'smoothstep' },
  { id: 'e6', source: 'lgbm', target: 'stacking', animated: true },
  { id: 'e7', source: 'catboost', target: 'stacking', animated: true },
  { id: 'e8', source: 'xgboost', target: 'stacking', animated: true },
  { id: 'e9', source: 'stacking', target: 'artifacts', animated: true },
];

const TrainerArchitectureDiagram = () => {
  return (
    <div className="w-full flex justify-center">
      <div className="h-[400px] w-full max-w-5xl bg-white rounded-xl shadow border border-gray-200">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          fitView
          panOnDrag={false}
          zoomOnScroll={false}
          zoomOnPinch={false}
          zoomOnDoubleClick={false}
          nodesDraggable={false}
          nodesConnectable={false}
          elementsSelectable={false}
        >
          <Background color="#e0f2fe" gap={24} />
          <Controls showInteractive={false} />
        </ReactFlow>
      </div>
    </div>
  );
};

export default TrainerArchitectureDiagram; 