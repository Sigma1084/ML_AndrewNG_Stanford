<!DOCTYPE html>
<html>

<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Week11</title>
  <link rel="stylesheet" href="https://stackedit.io/style.css" />
</head>

<body class="stackedit">
  <div class="stackedit__left">
    <div class="stackedit__toc">
      
<ul>
<li><a href="#application-example">Application Example</a></li>
<li><a href="#photo-ocr-optical-character-recognition">Photo OCR (Optical Character Recognition)</a>
<ul>
<li><a href="#pipeline">Pipeline</a></li>
<li><a href="#sliding-windows">Sliding Windows</a></li>
<li><a href="#getting-lots-of-data-artificial-data-synthesis">Getting Lots of Data: Artificial Data Synthesis</a></li>
<li><a href="#ceiling-analysis-what-part-on-the-pipeline-to-work-on-next">Ceiling Analysis (What part on the pipeline to work on next)</a></li>
</ul>
</li>
</ul>

    </div>
  </div>
  <div class="stackedit__right">
    <div class="stackedit__html">
      <h1 id="application-example">Application Example</h1>
<h1 id="photo-ocr-optical-character-recognition">Photo OCR (Optical Character Recognition)</h1>
<h2 id="pipeline">Pipeline</h2>
<ul>
<li>Text Detection (From an image)</li>
<li>Character Segmentation</li>
<li>Character Classification</li>
</ul>
<pre class=" language-mermaid"><svg id="mermaid-svg-nKiXEThfjcHwBhfT" width="100%" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" height="127.7125015258789" style="max-width: 804.5px;" viewBox="0 0 804.5 127.7125015258789"><style>#mermaid-svg-nKiXEThfjcHwBhfT{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;fill:#000000;}#mermaid-svg-nKiXEThfjcHwBhfT .error-icon{fill:#552222;}#mermaid-svg-nKiXEThfjcHwBhfT .error-text{fill:#552222;stroke:#552222;}#mermaid-svg-nKiXEThfjcHwBhfT .edge-thickness-normal{stroke-width:2px;}#mermaid-svg-nKiXEThfjcHwBhfT .edge-thickness-thick{stroke-width:3.5px;}#mermaid-svg-nKiXEThfjcHwBhfT .edge-pattern-solid{stroke-dasharray:0;}#mermaid-svg-nKiXEThfjcHwBhfT .edge-pattern-dashed{stroke-dasharray:3;}#mermaid-svg-nKiXEThfjcHwBhfT .edge-pattern-dotted{stroke-dasharray:2;}#mermaid-svg-nKiXEThfjcHwBhfT .marker{fill:#666;stroke:#666;}#mermaid-svg-nKiXEThfjcHwBhfT .marker.cross{stroke:#666;}#mermaid-svg-nKiXEThfjcHwBhfT svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;}#mermaid-svg-nKiXEThfjcHwBhfT .label{font-family:"trebuchet ms",verdana,arial,sans-serif;color:#000000;}#mermaid-svg-nKiXEThfjcHwBhfT .cluster-label text{fill:#333;}#mermaid-svg-nKiXEThfjcHwBhfT .cluster-label span{color:#333;}#mermaid-svg-nKiXEThfjcHwBhfT .label text,#mermaid-svg-nKiXEThfjcHwBhfT span{fill:#000000;color:#000000;}#mermaid-svg-nKiXEThfjcHwBhfT .node rect,#mermaid-svg-nKiXEThfjcHwBhfT .node circle,#mermaid-svg-nKiXEThfjcHwBhfT .node ellipse,#mermaid-svg-nKiXEThfjcHwBhfT .node polygon,#mermaid-svg-nKiXEThfjcHwBhfT .node path{fill:#eee;stroke:#999;stroke-width:1px;}#mermaid-svg-nKiXEThfjcHwBhfT .node .label{text-align:center;}#mermaid-svg-nKiXEThfjcHwBhfT .node.clickable{cursor:pointer;}#mermaid-svg-nKiXEThfjcHwBhfT .arrowheadPath{fill:#333333;}#mermaid-svg-nKiXEThfjcHwBhfT .edgePath .path{stroke:#666;stroke-width:1.5px;}#mermaid-svg-nKiXEThfjcHwBhfT .flowchart-link{stroke:#666;fill:none;}#mermaid-svg-nKiXEThfjcHwBhfT .edgeLabel{background-color:white;text-align:center;}#mermaid-svg-nKiXEThfjcHwBhfT .edgeLabel rect{opacity:0.5;background-color:white;fill:white;}#mermaid-svg-nKiXEThfjcHwBhfT .cluster rect{fill:hsl(210,66.6666666667%,95%);stroke:#26a;stroke-width:1px;}#mermaid-svg-nKiXEThfjcHwBhfT .cluster text{fill:#333;}#mermaid-svg-nKiXEThfjcHwBhfT .cluster span{color:#333;}#mermaid-svg-nKiXEThfjcHwBhfT div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:12px;background:hsl(-160,0%,93.3333333333%);border:1px solid #26a;border-radius:2px;pointer-events:none;z-index:100;}#mermaid-svg-nKiXEThfjcHwBhfT:root{--mermaid-font-family:"trebuchet ms",verdana,arial,sans-serif;}#mermaid-svg-nKiXEThfjcHwBhfT flowchart-v2{fill:apa;}</style><g transform="translate(0, 0)"><marker id="flowchart-pointEnd" class="marker flowchart" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="12" markerHeight="12" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker><marker id="flowchart-pointStart" class="marker flowchart" viewBox="0 0 10 10" refX="0" refY="5" markerUnits="userSpaceOnUse" markerWidth="12" markerHeight="12" orient="auto"><path d="M 0 5 L 10 10 L 10 0 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker><marker id="flowchart-circleEnd" class="marker flowchart" viewBox="0 0 10 10" refX="11" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"></circle></marker><marker id="flowchart-circleStart" class="marker flowchart" viewBox="0 0 10 10" refX="-1" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"></circle></marker><marker id="flowchart-crossEnd" class="marker cross flowchart" viewBox="0 0 11 11" refX="12" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"></path></marker><marker id="flowchart-crossStart" class="marker cross flowchart" viewBox="0 0 11 11" refX="-1" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"></path></marker><g class="root"><g class="clusters"><g class="cluster default" id="Pipeline"><rect style="" rx="0" ry="0" x="167.4749984741211" y="8" width="629.0249938964844" height="111.7125015258789"></rect><g class="cluster-label" transform="translate(453.5187454223633, 13)"><foreignObject width="56.9375" height="26.712499618530273"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;"><span class="nodeLabel">Pipeline</span></div></foreignObject></g></g><g class="cluster default" id="Input"><rect style="" rx="0" ry="0" x="8" y="8" width="109.4749984741211" height="111.7125015258789"></rect><g class="cluster-label" transform="translate(44.943748474121094, 13)"><foreignObject width="35.587501525878906" height="26.712499618530273"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;"><span class="nodeLabel">Input</span></div></foreignObject></g></g></g><g class="edgePaths"><path d="M92.4749984741211,63.85625076293945L96.64166514078777,63.85625076293945C100.80833180745442,63.85625076293945,109.14166514078777,63.85625076293945,117.4749984741211,63.85625076293945C125.80833180745442,63.85625076293945,134.14166514078775,63.85625076293945,142.4749984741211,63.85625076293945C150.80833180745444,63.85625076293945,159.14166514078775,63.85625076293945,167.4749984741211,63.85625076293945C175.80833180745444,63.85625076293945,184.14166514078775,63.85625076293945,188.30833180745444,63.85625076293945L192.4749984741211,63.85625076293945" id="L-A-B" class=" edge-thickness-normal edge-pattern-solid flowchart-link LS-A LE-B" style="fill:none;" marker-end="url(#flowchart-pointEnd)"></path><path d="M308.8624954223633,63.85625076293945L313.02916208902997,63.85625076293945C317.1958287556966,63.85625076293945,325.52916208902997,63.85625076293945,333.8624954223633,63.85625076293945C342.1958287556966,63.85625076293945,350.52916208902997,63.85625076293945,354.6958287556966,63.85625076293945L358.8624954223633,63.85625076293945" id="L-B-C" class=" edge-thickness-normal edge-pattern-solid flowchart-link LS-B LE-C" style="fill:none;" marker-end="url(#flowchart-pointEnd)"></path><path d="M547.2999954223633,63.85625076293945L551.4666620890299,63.85625076293945C555.6333287556967,63.85625076293945,563.9666620890299,63.85625076293945,572.2999954223633,63.85625076293945C580.6333287556967,63.85625076293945,588.9666620890299,63.85625076293945,593.1333287556967,63.85625076293945L597.2999954223633,63.85625076293945" id="L-C-D" class=" edge-thickness-normal edge-pattern-solid flowchart-link LS-C LE-D" style="fill:none;" marker-end="url(#flowchart-pointEnd)"></path></g><g class="edgeLabels"><g class="edgeLabel"><g class="label" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;"><span class="edgeLabel"></span></div></foreignObject></g></g></g><g class="nodes"><g class="node default default" id="flowchart-B-201" transform="translate(250.6687469482422, 63.85625076293945)"><rect class="basic label-container" style="" rx="5" ry="5" x="-58.19375228881836" y="-20.856249809265137" width="116.38750457763672" height="41.71249961853027"></rect><g class="label" style="" transform="translate(-50.69375228881836, -13.356249809265137)"><foreignObject width="101.38750457763672" height="26.712499618530273"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;"><span class="nodeLabel">Text Detection</span></div></foreignObject></g></g><g class="node default default" id="flowchart-C-203" transform="translate(453.0812454223633, 63.85625076293945)"><rect class="basic label-container" style="" rx="5" ry="5" x="-94.21875" y="-20.856249809265137" width="188.4375" height="41.71249961853027"></rect><g class="label" style="" transform="translate(-86.71875, -13.356249809265137)"><foreignObject width="173.4375" height="26.712499618530273"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;"><span class="nodeLabel">Character  Segmentation</span></div></foreignObject></g></g><g class="node default default" id="flowchart-D-205" transform="translate(684.3999938964844, 63.85625076293945)"><rect class="basic label-container" style="" rx="5" ry="5" x="-87.0999984741211" y="-20.856249809265137" width="174.1999969482422" height="41.71249961853027"></rect><g class="label" style="" transform="translate(-79.5999984741211, -13.356249809265137)"><foreignObject width="159.1999969482422" height="26.712499618530273"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;"><span class="nodeLabel">Character  Recognition</span></div></foreignObject></g></g><g class="node default default" id="flowchart-A-199" transform="translate(62.73749923706055, 63.85625076293945)"><rect class="basic label-container" style="" rx="5" ry="5" x="-29.73750114440918" y="-20.856249809265137" width="59.47500228881836" height="41.71249961853027"></rect><g class="label" style="" transform="translate(-22.23750114440918, -13.356249809265137)"><foreignObject width="44.47500228881836" height="26.712499618530273"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;"><span class="nodeLabel">Image</span></div></foreignObject></g></g></g></g></g></svg></pre>
<h2 id="sliding-windows">Sliding Windows</h2>
<h3 id="text-detection">Text Detection</h3>
<p><img src="https://drive.google.com/uc?id=1MobxuqMALMNR87ffAmFXWfikhCbs_02h" alt="text_detection"></p>
<h3 id="d-sliding-window-for-character-segmentation">1D Sliding Window for character segmentation</h3>
<p><img src="https://drive.google.com/uc?id=1_YixSyy-vOk0O49fHOw5XKeJHGmJdKjy" alt="sliding_window"></p>
<h2 id="getting-lots-of-data-artificial-data-synthesis">Getting Lots of Data: Artificial Data Synthesis</h2>
<ul>
<li>Manually create data or manually labelling a lot of training examples</li>
<li>Introducing distortions in the existing data and getting more examples out of the current examples</li>
</ul>
<ol>
<li>Make sure you have a low bias classifier before expending the effort. (Plot learning curves). E.g. keep increasing the number of features/number of hidden units in neural network until you have a low bias classifier.</li>
<li>Important Question: “How much work would it be to get 10x as much data as we  currently have?”
<ul>
<li>Artificial Data Synthesis</li>
<li>Collect / Label it manually</li>
<li>“Crowd Source” (eg. Amazon Mechanical Turk)</li>
</ul>
</li>
</ol>
<h2 id="ceiling-analysis-what-part-on-the-pipeline-to-work-on-next">Ceiling Analysis (What part on the pipeline to work on next)</h2>
<p>Consider the previous pipeline</p>
<pre class=" language-mermaid"><svg id="mermaid-svg-aPWkli3NJmOos78l" width="100%" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" height="127.7125015258789" style="max-width: 804.5px;" viewBox="0 0 804.5 127.7125015258789"><style>#mermaid-svg-aPWkli3NJmOos78l{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;fill:#000000;}#mermaid-svg-aPWkli3NJmOos78l .error-icon{fill:#552222;}#mermaid-svg-aPWkli3NJmOos78l .error-text{fill:#552222;stroke:#552222;}#mermaid-svg-aPWkli3NJmOos78l .edge-thickness-normal{stroke-width:2px;}#mermaid-svg-aPWkli3NJmOos78l .edge-thickness-thick{stroke-width:3.5px;}#mermaid-svg-aPWkli3NJmOos78l .edge-pattern-solid{stroke-dasharray:0;}#mermaid-svg-aPWkli3NJmOos78l .edge-pattern-dashed{stroke-dasharray:3;}#mermaid-svg-aPWkli3NJmOos78l .edge-pattern-dotted{stroke-dasharray:2;}#mermaid-svg-aPWkli3NJmOos78l .marker{fill:#666;stroke:#666;}#mermaid-svg-aPWkli3NJmOos78l .marker.cross{stroke:#666;}#mermaid-svg-aPWkli3NJmOos78l svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;}#mermaid-svg-aPWkli3NJmOos78l .label{font-family:"trebuchet ms",verdana,arial,sans-serif;color:#000000;}#mermaid-svg-aPWkli3NJmOos78l .cluster-label text{fill:#333;}#mermaid-svg-aPWkli3NJmOos78l .cluster-label span{color:#333;}#mermaid-svg-aPWkli3NJmOos78l .label text,#mermaid-svg-aPWkli3NJmOos78l span{fill:#000000;color:#000000;}#mermaid-svg-aPWkli3NJmOos78l .node rect,#mermaid-svg-aPWkli3NJmOos78l .node circle,#mermaid-svg-aPWkli3NJmOos78l .node ellipse,#mermaid-svg-aPWkli3NJmOos78l .node polygon,#mermaid-svg-aPWkli3NJmOos78l .node path{fill:#eee;stroke:#999;stroke-width:1px;}#mermaid-svg-aPWkli3NJmOos78l .node .label{text-align:center;}#mermaid-svg-aPWkli3NJmOos78l .node.clickable{cursor:pointer;}#mermaid-svg-aPWkli3NJmOos78l .arrowheadPath{fill:#333333;}#mermaid-svg-aPWkli3NJmOos78l .edgePath .path{stroke:#666;stroke-width:1.5px;}#mermaid-svg-aPWkli3NJmOos78l .flowchart-link{stroke:#666;fill:none;}#mermaid-svg-aPWkli3NJmOos78l .edgeLabel{background-color:white;text-align:center;}#mermaid-svg-aPWkli3NJmOos78l .edgeLabel rect{opacity:0.5;background-color:white;fill:white;}#mermaid-svg-aPWkli3NJmOos78l .cluster rect{fill:hsl(210,66.6666666667%,95%);stroke:#26a;stroke-width:1px;}#mermaid-svg-aPWkli3NJmOos78l .cluster text{fill:#333;}#mermaid-svg-aPWkli3NJmOos78l .cluster span{color:#333;}#mermaid-svg-aPWkli3NJmOos78l div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:12px;background:hsl(-160,0%,93.3333333333%);border:1px solid #26a;border-radius:2px;pointer-events:none;z-index:100;}#mermaid-svg-aPWkli3NJmOos78l:root{--mermaid-font-family:"trebuchet ms",verdana,arial,sans-serif;}#mermaid-svg-aPWkli3NJmOos78l flowchart-v2{fill:apa;}</style><g transform="translate(0, 0)"><marker id="flowchart-pointEnd" class="marker flowchart" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="12" markerHeight="12" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker><marker id="flowchart-pointStart" class="marker flowchart" viewBox="0 0 10 10" refX="0" refY="5" markerUnits="userSpaceOnUse" markerWidth="12" markerHeight="12" orient="auto"><path d="M 0 5 L 10 10 L 10 0 z" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"></path></marker><marker id="flowchart-circleEnd" class="marker flowchart" viewBox="0 0 10 10" refX="11" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"></circle></marker><marker id="flowchart-circleStart" class="marker flowchart" viewBox="0 0 10 10" refX="-1" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><circle cx="5" cy="5" r="5" class="arrowMarkerPath" style="stroke-width: 1; stroke-dasharray: 1, 0;"></circle></marker><marker id="flowchart-crossEnd" class="marker cross flowchart" viewBox="0 0 11 11" refX="12" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"></path></marker><marker id="flowchart-crossStart" class="marker cross flowchart" viewBox="0 0 11 11" refX="-1" refY="5.2" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M 1,1 l 9,9 M 10,1 l -9,9" class="arrowMarkerPath" style="stroke-width: 2; stroke-dasharray: 1, 0;"></path></marker><g class="root"><g class="clusters"><g class="cluster default" id="Pipeline"><rect style="" rx="0" ry="0" x="167.4749984741211" y="8" width="629.0249938964844" height="111.7125015258789"></rect><g class="cluster-label" transform="translate(453.5187454223633, 13)"><foreignObject width="56.9375" height="26.712499618530273"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;"><span class="nodeLabel">Pipeline</span></div></foreignObject></g></g><g class="cluster default" id="Input"><rect style="" rx="0" ry="0" x="8" y="8" width="109.4749984741211" height="111.7125015258789"></rect><g class="cluster-label" transform="translate(44.943748474121094, 13)"><foreignObject width="35.587501525878906" height="26.712499618530273"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;"><span class="nodeLabel">Input</span></div></foreignObject></g></g></g><g class="edgePaths"><path d="M92.4749984741211,63.85625076293945L96.64166514078777,63.85625076293945C100.80833180745442,63.85625076293945,109.14166514078777,63.85625076293945,117.4749984741211,63.85625076293945C125.80833180745442,63.85625076293945,134.14166514078775,63.85625076293945,142.4749984741211,63.85625076293945C150.80833180745444,63.85625076293945,159.14166514078775,63.85625076293945,167.4749984741211,63.85625076293945C175.80833180745444,63.85625076293945,184.14166514078775,63.85625076293945,188.30833180745444,63.85625076293945L192.4749984741211,63.85625076293945" id="L-A-B" class=" edge-thickness-normal edge-pattern-solid flowchart-link LS-A LE-B" style="fill:none;" marker-end="url(#flowchart-pointEnd)"></path><path d="M308.8624954223633,63.85625076293945L313.02916208902997,63.85625076293945C317.1958287556966,63.85625076293945,325.52916208902997,63.85625076293945,333.8624954223633,63.85625076293945C342.1958287556966,63.85625076293945,350.52916208902997,63.85625076293945,354.6958287556966,63.85625076293945L358.8624954223633,63.85625076293945" id="L-B-C" class=" edge-thickness-normal edge-pattern-solid flowchart-link LS-B LE-C" style="fill:none;" marker-end="url(#flowchart-pointEnd)"></path><path d="M547.2999954223633,63.85625076293945L551.4666620890299,63.85625076293945C555.6333287556967,63.85625076293945,563.9666620890299,63.85625076293945,572.2999954223633,63.85625076293945C580.6333287556967,63.85625076293945,588.9666620890299,63.85625076293945,593.1333287556967,63.85625076293945L597.2999954223633,63.85625076293945" id="L-C-D" class=" edge-thickness-normal edge-pattern-solid flowchart-link LS-C LE-D" style="fill:none;" marker-end="url(#flowchart-pointEnd)"></path></g><g class="edgeLabels"><g class="edgeLabel"><g class="label" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;"><span class="edgeLabel"></span></div></foreignObject></g></g><g class="edgeLabel"><g class="label" transform="translate(0, 0)"><foreignObject width="0" height="0"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;"><span class="edgeLabel"></span></div></foreignObject></g></g></g><g class="nodes"><g class="node default default" id="flowchart-B-217" transform="translate(250.6687469482422, 63.85625076293945)"><rect class="basic label-container" style="" rx="5" ry="5" x="-58.19375228881836" y="-20.856249809265137" width="116.38750457763672" height="41.71249961853027"></rect><g class="label" style="" transform="translate(-50.69375228881836, -13.356249809265137)"><foreignObject width="101.38750457763672" height="26.712499618530273"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;"><span class="nodeLabel">Text Detection</span></div></foreignObject></g></g><g class="node default default" id="flowchart-C-219" transform="translate(453.0812454223633, 63.85625076293945)"><rect class="basic label-container" style="" rx="5" ry="5" x="-94.21875" y="-20.856249809265137" width="188.4375" height="41.71249961853027"></rect><g class="label" style="" transform="translate(-86.71875, -13.356249809265137)"><foreignObject width="173.4375" height="26.712499618530273"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;"><span class="nodeLabel">Character  Segmentation</span></div></foreignObject></g></g><g class="node default default" id="flowchart-D-221" transform="translate(684.3999938964844, 63.85625076293945)"><rect class="basic label-container" style="" rx="5" ry="5" x="-87.0999984741211" y="-20.856249809265137" width="174.1999969482422" height="41.71249961853027"></rect><g class="label" style="" transform="translate(-79.5999984741211, -13.356249809265137)"><foreignObject width="159.1999969482422" height="26.712499618530273"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;"><span class="nodeLabel">Character  Recognition</span></div></foreignObject></g></g><g class="node default default" id="flowchart-A-215" transform="translate(62.73749923706055, 63.85625076293945)"><rect class="basic label-container" style="" rx="5" ry="5" x="-29.73750114440918" y="-20.856249809265137" width="59.47500228881836" height="41.71249961853027"></rect><g class="label" style="" transform="translate(-22.23750114440918, -13.356249809265137)"><foreignObject width="44.47500228881836" height="26.712499618530273"><div xmlns="http://www.w3.org/1999/xhtml" style="display: inline-block; white-space: nowrap;"><span class="nodeLabel">Image</span></div></foreignObject></g></g></g></g></g></svg></pre>
<p>Calculating the performance by making components perfect manually, suppose we see</p>

<table>
<thead>
<tr>
<th>Component</th>
<th>Accuracy</th>
</tr>
</thead>
<tbody>
<tr>
<td>Overall System</td>
<td>72%</td>
</tr>
<tr>
<td>Text Detection</td>
<td>89%</td>
</tr>
<tr>
<td>Character Segmentation</td>
<td>90%</td>
</tr>
<tr>
<td>Character Recognition</td>
<td>100%</td>
</tr>
</tbody>
</table><ul>
<li>By having a perfect Text Detection, the accuracy went up by 17%</li>
<li>But after Text Detection, when Character Segmentation was perfected, the accuracy went up only by 1%</li>
<li>Again, perfect Character Recognition gave a boost of 10%</li>
</ul>
<p>This means, that having a perfect character segmentation can only increase the accuracy by 1%.<br>
So, putting more work on Text Detection or Character Recognition is better!</p>

    </div>
  </div>
</body>

</html>