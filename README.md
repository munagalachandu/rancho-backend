L1
using UnityEngine;

public class Lab1 : MonoBehaviour
{
    public Animator animator;
    public ParticleSystem ps;

    void Start()
    {
        animator = GetComponent<Animator>();
        ps = GetComponentInChildren<ParticleSystem>();
    }

    void Update()
    {
        // Animation A (Key 1)
        if (Input.GetKeyDown(KeyCode.Alpha1))
            animator.SetBool("a", true);

        if (Input.GetKeyUp(KeyCode.Alpha1))
            animator.SetBool("a", false);

        // Animation B (Key 2)
        if (Input.GetKeyDown(KeyCode.Alpha2))
            animator.SetBool("b", true);

        if (Input.GetKeyUp(KeyCode.Alpha2))
            animator.SetBool("b", false);

        // Particle toggle (Key P)
        if (Input.GetKeyDown(KeyCode.P))
        {
            if (ps.isPlaying)
                ps.Stop();
            else
                ps.Play();
        }
    }
}

L2
using UnityEngine;

public class Lab1 : MonoBehaviour
{
    public Animator animator;
    public ParticleSystem ps;

    void Start()
    {
        animator = GetComponent<Animator>();
        ps = GetComponentInChildren<ParticleSystem>();
    }

    void Update()
    {
        // Animation A (Key 1)
        if (Input.GetKeyDown(KeyCode.Alpha1))
            animator.SetBool("a", true);

        if (Input.GetKeyUp(KeyCode.Alpha1))
            animator.SetBool("a", false);

        // Animation B (Key 2)
        if (Input.GetKeyDown(KeyCode.Alpha2))
            animator.SetBool("b", true);

        if (Input.GetKeyUp(KeyCode.Alpha2))
            animator.SetBool("b", false);

        // Particle toggle (Key P)
        if (Input.GetKeyDown(KeyCode.P))
        {
            if (ps.isPlaying)
                ps.Stop();
            else
                ps.Play();
        }
    }
}

L3
using UnityEngine;
using System.Collections;
using System.Collections.Generic;

public class Lab2 : MonoBehaviour
{
    public float speed = 5f;

    void Update()
    {
        // Get input (WASD / Arrow keys)
        float moveX = Input.GetAxis("Horizontal");
        float moveZ = Input.GetAxis("Vertical");

        // Create movement vector
        Vector3 move = new Vector3(moveX, 0f, moveZ);

        // Move player
        transform.Translate(move * speed * Time.deltaTime);
    }

    void OnCollisionEnter(Collision collision)
    {
        // Check if object is obstacle
        if (collision.gameObject.CompareTag("Obstacle"))
        {
            Debug.Log("Obstacle Found!");
        }
    }
}

L4
using UnityEngine;
using UnityEngine.AI;

public class Lab3 : MonoBehaviour
{
    public Transform player;           // Player reference
    public float roamRadius = 10f;     // Random roaming area
    public float detectionRange = 8f;  // Distance to detect player

    NavMeshAgent agent;
    Vector3 roamPoint;

    void Start()
    {
        agent = GetComponent<NavMeshAgent>();
        SetNewRoamPoint(); // start roaming
    }

    void Update()
    {
        float distance = Vector3.Distance(transform.position, player.position);

        // If player is near → follow
        if (distance <= detectionRange)
        {
            agent.SetDestination(player.position);
        }
        else
        {
            // If reached roam point → pick new one
            if (!agent.pathPending && agent.remainingDistance < 0.5f)
            {
                SetNewRoamPoint();
            }
        }
    }

    void SetNewRoamPoint()
    {
        Vector3 randomDirection = Random.insideUnitSphere * roamRadius;
        randomDirection += transform.position;

        NavMeshHit hit;
        NavMesh.SamplePosition(randomDirection, out hit, roamRadius, 1);

        roamPoint = hit.position;
        agent.SetDestination(roamPoint);
    }
}


<!DOCTYPE html>
<html>
<head>
  <title>A-Frame Hover & Click Reset</title>
  <script src="aframe.min.js"></script>
 
</head>

<body>
  <a-scene>

    <!-- Camera -->
    <a-camera position="0 1.6 4"
              cursor="rayOrigin: mouse"
              raycaster="objects: .interactive">
    </a-camera>

    <!-- Lights -->
    <a-light type="directional" position="1 2 1" intensity="0.9"></a-light>
    <a-light type="ambient" intensity="0.6"></a-light>

  <!-- Left -->
<a-torus-knot class="interactive"
position="-2 1.2 -3"
radius="0.45"
radius-tubular="0.15"
material="color: #FF0000">
</a-torus-knot>

<!-- Center -->
<a-octahedron class="interactive"
position="0 1.2 0"
radius="0.6"
material="color: #38BDF8; emissive: #0b2a3a; metalness: 0.35; roughness: 0.25">
</a-octahedron>

<!-- Right -->
<a-torus class="interactive"
position="2 1.2 -3"
radius="0.6"
radius-tubular="0.22"
material="color: #FFD700">
</a-torus>


    <!-- Ground -->
    <a-plane position="0 0 -4"
             rotation="-90 0 0"
             width="10"
             height="10"
             material="color: #7BC8A4">
    </a-plane>

  </a-scene>
</body>
</html>
------------------------
## Lab 5
=====
<!DOCTYPE html>
<html>
<head>
  <title>A-Frame Hover & Click Reset</title>
  <script src="aframe.min.js"></script>
  <script>
    AFRAME.registerComponent('hover-color', {
      init: function () {
        const el = this.el;
        const originalColor = el.getAttribute('material').color;
        el.addEventListener('mouseenter', () => {
          el.setAttribute('material', 'color', '#FF0000'); 
        });
        el.addEventListener('mouseleave', () => {
          el.setAttribute('material', 'color', originalColor);
        });
      }
    });

    
    // Click scale toggle for BOX
    AFRAME.registerComponent('click-2', {
      init: function () {
        const el = this.el;
        let scaled = false;
        const originalColor = el.getAttribute('material').color;
        el.addEventListener('click', () => {
          if (!scaled) {
            el.setAttribute('material', 'color', '#0000FF'); 
          } else {
            el.setAttribute('material', 'color', originalColor);
          }
          scaled = !scaled;
        });
      }
    });
    AFRAME.registerComponent('click-scale-toggle', {
      init: function () {
        const el = this.el;
        let scaled = false;

        el.addEventListener('click', () => {
          if (!scaled) {
            el.setAttribute('scale', '2 2 2');
          } else {
            el.setAttribute('scale', '1 1 1');
          }
          scaled = !scaled;
        });
      }
    });
  </script>
</head>

<body>
  <a-scene>

    <!-- Camera -->
    <a-camera position="0 1.6 4"
              cursor="rayOrigin: mouse"
              raycaster="objects: .interactive">
    </a-camera>

    <!-- Lights -->
    <a-light type="directional" position="1 2 1" intensity="0.9"></a-light>
    <a-light type="ambient" intensity="0.6"></a-light>

    <!-- Cube: click to scale and reset -->
    <a-cone class="interactive"
           position="-1.8 1 -3"
           material="color: #FF0000"
           click-scale-toggle>
    </a-cone>

    <!-- Cylinder: hover color change and reset -->
    <a-cylinder class="interactive"
                position="0 1 -3"
                radius="0.5"
                height="1.5"
                material="color: #00FF00"
                hover-color>
    </a-cylinder>

    <!-- Torus: rotating -->
    <a-torus class="interactive"
    position="1.8 1.2 -3"
             radius="0.7"
             radius-tubular="0.25"
             material="color: #FFD700"
             animation="property: rotation;
                        to: 0 360 0;
                        loop: true;
                        dur: 2000"
            click-2>
    </a-torus>

    <!-- Ground -->
    <a-plane position="0 0 -4"
             rotation="-90 0 0"
             width="10"
             height="10"
             material="color: #7BC8A4">
    </a-plane>

  </a-scene>
</body>
</html>
------------------------------
=========
##Lab 6
<!DOCTYPE html>
<html>
<head>
  <title>360 Image Galery</title>

  <!-- LOCAL A-Frame -->
  <script src="aframe.min.js"></script>

  <script>
    // Component to change 360 image on hotspot click
    AFRAME.registerComponent('change-scene', {
      schema: { target: {type: 'string'} },
      init: function () {
        this.el.addEventListener('click', () => {
          document.querySelector('#sky').setAttribute('src', this.data.target);
        });
      }
    });
  </script>
</head>

<body style="margin:0; overflow:hidden;">

<a-scene>

  <!-- Load LOCAL images -->
  <a-assets>
    <img id="img1" src="images/img1.jpg">
    <img id="img2" src="images/img2.jpg">
    <img id="img3" src="images/img3.jpg">
  </a-assets>

  <!-- Camera with gyroscope and mouse support -->
  <a-camera position="0 1.6 0" look-controls cursor="rayOrigin: mouse" raycaster="objects: .clickable">
  </a-camera>

  <!-- 360 Sky -->
  <a-sky id="sky" src="#img1" rotation="0 -90 0"></a-sky>

  <!-- Hotspot 1 -->
  <a-circle class="clickable"
            position="-2 1.5 -3"
            radius="0.3"
            material="color: red"
            change-scene="target: #img1">
  </a-circle>
  <!-- Label for Hotspot 1 -->
  <a-text value="Living Room"
          position="-2 2.1 -3"
          color="white"
          align="center"
          width="3">
  </a-text>

  <!-- Hotspot 2 -->
  <a-circle class="clickable"
            position="0 1.5 -3"
            radius="0.3"
            material="color: blue"
            change-scene="target: #img2">
  </a-circle>
  <!-- Label for Hotspot 2 -->
  <a-text value="Kitchen"
          position="0 2.1 -3"
          color="white"
          align="center"
          width="3">
  </a-text>

  <!-- Hotspot 3 -->
  <a-circle class="clickable"
            position="2 1.5 -3"
            radius="0.3"
            material="color: green"
            change-scene="target: #img3">
  </a-circle>
  <!-- Label for Hotspot 3 -->
  <a-text value="Bedroom"
          position="2 2.1 -3"
          color="white"
          align="center"
          width="3">
  </a-text>

</a-scene>

</body>
</html>
------------------------------
<!doctype html>
<html>
<head>
<title>A-Frame Geolocation</title>

<script src="/aframe.min.js"></script>
<script>
document.addEventListener('DOMContentLoaded', function() {
function showShapes(position)
 {
var currentLatitude = position.coords.latitude;
var currentLongitude = position.coords.longitude;
console.log("Latitude: " + currentLatitude);
console.log("Longitude: " + currentLongitude);
document.getElementById('currentLocation').innerHTML =
  `Latitude: ${currentLatitude.toFixed(6)}, Longitude: ${currentLongitude.toFixed(6)}`;

var locations = [
{ id: "box", lat:12.9564672,lon:77.6208384, threshold: 0.05 },
{ id: "cylinder", lat: 12.90509057, lon: 77.55971556, threshold: 0.05 },
{ id: "sphere", lat: 12.9564672,lon: 77.594624,threshold: 0.05}
];
locations.forEach(location => {
var shape = document.querySelector(`#${location.id}`);
if (shape && Math.abs(currentLatitude - location.lat) < location.threshold &&
Math.abs(currentLongitude - location.lon) < location.threshold) {
shape.setAttribute('visible', true);
}
});
}
function locationError(error) {
console.error("Error getting location: ", error);
document.getElementById('currentLocation').innerHTML =
`Error getting location: ${error.message}`;
}
function getLocation() {
if (navigator.geolocation) {
navigator.geolocation.getCurrentPosition(showShapes, locationError, { enableHighAccuracy: true, timeout: 10000, maximumAge: 0 });
} else {
document.getElementById('currentLocation').innerHTML =
"Geolocation is not supported by this browser.";
}
}
getLocation();
});</script>
</head>
<body><h3 id="currentLocation">Fetching location...</h3>
    <a-scene>

        <!-- Glowing Red Box -->
        <a-box id="box"
               position="0 1 -4"
               rotation="0 45 0"
               depth="1.2"
               height="1.2"
               width="1.2"
               material="color: #FF3B3B; emissive: #551111; metalness: 0.3; roughness: 0.4"
               visible="false">
        </a-box>
      
        <a-torus-knot id="cylinder"
        position="2.5 1.2 -4"
        radius="0.7"
        radius-tubular="0.2"
        material="color: #3B82F6; emissive: #0B1E3A; metalness: 0.4; roughness: 0.2"
        visible="false">
</a-torus-knot>



        <!-- Bright Green Sphere -->
        <a-sphere id="sphere"
                  position="-2.5 1.2 -4"
                  radius="1"
                  material="color: #22C55E; emissive: #113322; metalness: 0.25; roughness: 0.35"
                  visible="false">
        </a-sphere>
      
        <!-- Lighting for better visuals -->
        <a-light type="ambient" intensity="0.6"></a-light>
        <a-light type="directional" position="2 4 2" intensity="0.8"></a-light>
      
        <a-camera gps-camera rotation-reader></a-camera>
      
      </a-scene>
      
</body>
</html>
-----------------------------------
<!DOCTYPE html>
<html>
<head>
  <title>A-Frame AR Marker</title>

  <script src="https://aframe.io/releases/1.3.0/aframe.min.js"></script>
  <script src="https://cdn.jsdelivr.net/gh/AR-js-org/AR.js@3.4.5/aframe/build/aframe-ar.min.js"></script>
  
</head>

<body style="margin: 0; overflow: hidden;">
  <a-scene embedded arjs="sourceType: webcam; debugUIEnabled: false;">

    <!-- Marker -->
   
    <a-marker type="pattern" url="pattern-letterC.patt">

        <a-torus-knot position="0 1 0"
               material="color: #14B8A6; opacity: 1; ">
        </a-torus-kot>
      
      
      </a-marker>
      
    <!-- Camera -->
    <a-entity camera></a-entity>

  </a-scene>
</body>
</html>
------------------------------
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title> Marker Based</title>

  <script src="aframe.min.js"></script>
  <script src="mindar-image-aframe.prod.js"></script>

  <style>
    body { margin: 0; overflow: hidden; }
  </style>
</head>

<body>

<a-scene
  mindar-image="imageTargetSrc: targets.mind;"
  vr-mode-ui="false"
  device-orientation-permission-ui="false">

  <a-camera position="0 0 0" look-controls="false"></a-camera>

  <a-entity mindar-image-target="targetIndex: 0">
    <a-torus-knot
      radius="0.3"
      radius-tubular="0.05"
      color="#63e63d"
      animation="property: rotation; to: 0 360 0; loop: true; dur: 3000">
    </a-torus-knot>
  </a-entity>

</a-scene>

</body>
</html>
-------------------------------------
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Simple A-Frame Scene</title>
  <script src="aframe.min.js"></script>
  <style>
    /* Ensure the body takes full height */
    body, html {
      margin: 0;
      padding: 0;
      height: 100%;
    }

    /* Style the button to make it visible and positioned correctly */
    button {
      position: absolute;
      top: 20px;
      left: 50%;
      transform: translateX(-50%);
      padding: 10px 20px;
      font-size: 16px;
      background-color: #4b1bdd;
      border: none;
      color: white;
      border-radius: 5px;
      cursor: pointer;
      z-index: 10; /* Ensure it's above the scene */
    }

    /* Add hover effect for the button */
    button:hover {
      background-color: #bd2fff;
    }
  </style>
</head>
<body>
  <a-scene>
    <a-assets>
      <a-asset-item id="value" src="spi.glb"></a-asset-item>
    </a-assets>

    <a-entity position="0 1.6 0">
      <a-camera></a-camera>
    </a-entity>

    <a-entity id="model" gltf-model="#value" position="0 0 -5" rotation="0 45 0" scale="3 3 3"    light="type: directional; intensity: 20"   ></a-entity>
  </a-scene>

  <!-- Button to trigger rotation -->
  <button onclick="rotateModel()">Rotate Model 45°</button>

  <script>
    // Function to rotate the model by 45 degrees each time
    function rotateModel() {
      const model = document.querySelector('#model');
      
      // Get the current rotation
      let currentRotation = model.getAttribute('rotation');
      
      // Increment the Y rotation by 45 degrees
      currentRotation.y += 45;
      
      // Apply the updated rotation
      model.setAttribute('rotation', currentRotation);
    }
  </script>
</body>
</html>
--------------------------------
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>GLB with Marker (MindAR)</title>

  <!-- A-Frame -->
  <script src="aframe.min.js"></script>

  <!-- MindAR -->
  <script src="mindar-image-aframe.prod.js"></script>

  <style>
    body { margin: 0; overflow: hidden; }

    button {
      position: absolute;
      top: 20px;
      left: 50%;
      transform: translateX(-50%);
      padding: 10px 20px;
      font-size: 16px;
      background-color: #4b1bdd;
      border: none;
      color: white;
      border-radius: 5px;
      cursor: pointer;
      z-index: 10;
    }

    button:hover {
      background-color: #bd2fff;
    }
  </style>
</head>

<body>

  <a-scene
  mindar-image="imageTargetSrc: targets (1).mind;"
  vr-mode-ui="false"
  device-orientation-permission-ui="false"
  renderer="colorManagement: true; physicallyCorrectLights: true; exposure: 1.8"
>

  <!-- Assets -->
  <a-assets>
    <a-asset-item id="modelGLB" src="model.glb"></a-asset-item>
  </a-assets>

  <!-- 🔥 BEST LIGHTING SETUP -->

  <!-- Soft global light -->
  <a-light type="ambient" intensity="1.2" color="#ffffff"></a-light>

  <!-- Main light (front/top) -->
  <a-light type="directional" position="2 4 2" intensity="2"></a-light>

  <!-- Fill light (reduces harsh shadows) -->
  <a-light type="directional" position="-2 2 1" intensity="1"></a-light>

  <!-- Camera -->
  <a-camera position="0 0 0" look-controls="false"></a-camera>

  <!-- Marker Target -->
  <a-entity mindar-image-target="targetIndex: 0">

    <!-- GLB Model -->
    <a-entity
      id="model"
      gltf-model="#modelGLB"
      position="0 0 0"
      rotation="0 0 0"
      scale="1.5 1.5 1.5"
      material="metalness: 0; roughness: 1"
    >
    </a-entity>

  </a-entity>

</a-scene>

<!-- Rotate Button -->
<button onclick="rotateModel()">Rotate Model 45°</button>

<script>
function rotateModel() {
  const model = document.querySelector('#model');

  let currentRotation = model.getAttribute('rotation');

  currentRotation.y += 45;

  model.setAttribute('rotation', currentRotation);
}
</script>

</body>
</html>
