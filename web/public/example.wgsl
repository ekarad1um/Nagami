// Ported from https://www.shadertoy.com/view/MtX3Ws
// License Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.
// Created by S. Guillitte 2015

struct Inputs {
    size: vec2f,
    time: f32,
    mouse: vec4f,
}
@group(0) @binding(0) var<uniform> inputs: Inputs;

const zoom: f32 = 1.5;

fn complexSquare(a: vec2f) -> vec2f {
    return vec2f(a.x * a.x - a.y * a.y, 2.0 * a.x * a.y);
}

fn rotationMatrix(angle: f32) -> mat2x2f {
    let cosAngle = cos(angle);
    let sinAngle = sin(angle);
    return mat2x2f(cosAngle, sinAngle, -sinAngle, cosAngle);
}

fn intersectSphere(rayOrigin: vec3f, rayDirection: vec3f, sphere: vec4f) -> vec2f {
    let originToCenter = rayOrigin - sphere.xyz;
    let halfB = dot(originToCenter, rayDirection);
    let cVal = dot(originToCenter, originToCenter) - sphere.w * sphere.w;
    let discriminant = halfB * halfB - cVal;

    if (discriminant < 0.0) {
        return vec2f(-1.0);
    }

    let sqrtDiscriminant = sqrt(discriminant);
    return vec2f(-halfB - sqrtDiscriminant, -halfB + sqrtDiscriminant);
}

fn evaluateFractal(samplePoint: vec3f) -> f32 {
    var accumulatedDistance = 0.0;
    var point = samplePoint;
    let initialPoint = point;

    for (var i = 0u; i < 10u; i++) {
        point = 0.7 * abs(point) / dot(point, point) - 0.7;

        point = vec3f(point.x, complexSquare(point.yz)).zxy;
        accumulatedDistance += exp(-19.0 * abs(dot(point, initialPoint)));
    }

    return accumulatedDistance / 2.0;
}

fn performRaymarch(rayOrigin: vec3f, rayDirection: vec3f, entryExitDistances: vec2f) -> vec3f {
    var currentDistance = entryExitDistances.x;
    let stepSize = 0.02;

    var accumulatedColor = vec3f(0.0);
    var fractalValue = 0.0;

    for (var i = 0u; i < 64u; i++) {
        currentDistance += stepSize * exp(-2.0 * fractalValue);

        if (currentDistance > entryExitDistances.y) {
            break;
        }

        fractalValue = evaluateFractal(rayOrigin + currentDistance * rayDirection);
        accumulatedColor = 0.99 * accumulatedColor + 0.08 * vec3f(fractalValue * fractalValue * fractalValue, fractalValue * fractalValue, fractalValue);
    }

    return accumulatedColor;
}

@fragment
fn fragmentMain(@builtin(position) pos: vec4f) -> @location(0) vec4f {
    let time = inputs.time;
    let resolution = inputs.size.xy;
    let mouse = inputs.mouse;

    let fragmentCoordinate = vec2f(pos.x, resolution.y - pos.y);
    let normalizedCoords = fragmentCoordinate / resolution;

    var uv = -1.0 + 2.0 * normalizedCoords;
    uv.x *= resolution.x / resolution.y;

    var mouseUV = vec2f(0.0);
    if (mouse.z > 0.0) {
        mouseUV = (mouse.xy / resolution.xy) * 3.14159;
    }
    mouseUV -= 0.5;

    // --- Camera Setup ---
    var rayOrigin = zoom * vec3f(4.0);

    // Apply Y rotation
    rayOrigin = vec3f(rayOrigin.x, rayOrigin.yz * rotationMatrix(mouseUV.y));

    // Apply X rotation (with time)
    let rotatedXZ = rayOrigin.xz * rotationMatrix(mouseUV.x + 0.1 * time);
    rayOrigin = vec3f(rotatedXZ.x, rayOrigin.y, rotatedXZ.y);

    let lookTarget = vec3f(0.0);
    let forward = normalize(lookTarget - rayOrigin);
    let right = normalize(cross(forward, vec3f(0.0, 1.0, 0.0)));
    let up = normalize(cross(right, forward));
    let rayDirection = normalize(uv.x * right + uv.y * up + 4.0 * forward);

    let sphereIntersection = intersectSphere(rayOrigin, rayDirection, vec4f(0.0, 0.0, 0.0, 2.0));
    var finalColor = vec3f(0.0);

    // --- Raymarch ---
    // Optimization: Only run the heavy raymarch if the ray actually hits the sphere bounds
    if (sphereIntersection.x >= 0.0) {
        finalColor = performRaymarch(rayOrigin, rayDirection, sphereIntersection);
    }

    // --- Shade ---
    finalColor = 0.5 * log(1.0 + finalColor);

    return vec4f(finalColor, 1.0);
}
