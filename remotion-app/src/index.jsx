import React from 'react';
import {registerRoot, Composition, AbsoluteFill, useCurrentFrame, interpolate} from 'remotion';
const Hello = () => {
  const f = useCurrentFrame();
  const o = interpolate(f, [0, 20], [0, 1], {extrapolateRight: 'clamp'});
  return <AbsoluteFill style={{background: '#1a1040', color: '#fff', justifyContent: 'center', alignItems: 'center', fontSize: 80, opacity: o}}>content-ai</AbsoluteFill>;
};
registerRoot(() => <Composition id="Hello" component={Hello} durationInFrames={60} fps={30} width={720} height={1280} />);
