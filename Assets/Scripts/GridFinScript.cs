using UnityEngine;
using UnityEngine.InputSystem;

public class GridFin : MonoBehaviour
{
    [Header("Fin Settings")]
    [Tooltip("Maximum aerodynamic force this fin can produce in Newtons.")]
    public float maxFinForce = 50000f;

    [Tooltip("Fin surface area in m2 — real Falcon 9 is ~4x4 = 16m2.")]
    public float finArea = 16f;

    [Tooltip("Fin drag coefficient.")]
    public float finDragCoefficient = 1.5f;

    [Header("Deploy Settings")]
    [Tooltip("Altitude in metres at which fins automatically deploy.")]
    public float deployAltitude = 70000f;

    [Tooltip("Altitude in metres at which fins automatically retract.")]
    public float retractAltitude = 500f;

    [Tooltip("Speed in degrees per second that fins open and close.")]
    public float deploySpeed = 45f;

    [Tooltip("Angle when fully deployed in degrees.")]
    public float deployedAngle = 90f;

    [Tooltip("Angle when fully retracted in degrees.")]
    public float retractedAngle = 0f;

    [Header("Steering Settings")]
    [Tooltip("Maximum steering deflection angle in degrees.")]
    public float maxSteeringAngle = 30f;

    [Tooltip("Speed at which fin steers in degrees per second.")]
    public float steeringSpeed = 60f;

    [Header("Key Assignment")]
    [Tooltip("Key to tilt fin left.")]
    public FinKey leftKey = FinKey.Alpha1;

    [Tooltip("Key to tilt fin right.")]
    public FinKey rightKey = FinKey.Alpha2;

    [Header("Atmosphere")]
    public float seaLevelAirDensity = 1.225f;
    public float atmosphereScaleHeight = 8500f;

    public enum FinKey
    {
        Alpha1, Alpha2, Alpha3, Alpha4,
        Alpha5, Alpha6, Alpha7, Alpha8
    }

    // State
    private bool isDeployed = false;
    private float currentDeployAngle = 0f;
    private float currentSteeringAngle = 0f;
    private Rigidbody rb;
    private float currentForce = 0f;
    private float currentEffectiveArea = 0f;

    void Start()
    {
        rb = GetComponentInParent<Rigidbody>();

        if (rb == null)
        {
            Debug.LogError("GridFin on " + gameObject.name +
                           ": No Rigidbody found in parent!");
        }

        // Start fully retracted
        currentDeployAngle = retractedAngle;
        currentSteeringAngle = 0f;
        ApplyRotation();
    }

    void Update()
    {
        HandleDeployment();
        HandleSteering();
        AnimateFin();
    }

    void FixedUpdate()
    {
        if (isDeployed && rb != null)
        {
            ApplyAeroForce();
        }
        else
        {
            currentForce = 0f;
            currentEffectiveArea = 0f;
        }
    }

    void HandleDeployment()
    {
        float altitude = Mathf.Max(0f, transform.position.y);

        // Auto deploy when falling through deploy altitude
        if (!isDeployed && altitude <= deployAltitude
            && rb != null && rb.linearVelocity.y < 0f)
        {
            isDeployed = true;
            Debug.Log("GridFin " + gameObject.name + ": DEPLOYING");
        }

        // Auto retract near ground
        if (isDeployed && altitude <= retractAltitude)
        {
            isDeployed = false;
            Debug.Log("GridFin " + gameObject.name + ": RETRACTING");
        }
    }

    void HandleSteering()
    {
        // Only steer when deployed
        if (!isDeployed)
        {
            // Return to centre when not deployed
            currentSteeringAngle = Mathf.MoveTowards(
                currentSteeringAngle, 0f,
                steeringSpeed * Time.deltaTime
            );
            return;
        }

        bool leftHeld  = IsKeyHeld(leftKey);
        bool rightHeld = IsKeyHeld(rightKey);

        if (leftHeld && !rightHeld)
        {
            currentSteeringAngle = Mathf.MoveTowards(
                currentSteeringAngle,
                -maxSteeringAngle,
                steeringSpeed * Time.deltaTime
            );
        }
        else if (rightHeld && !leftHeld)
        {
            currentSteeringAngle = Mathf.MoveTowards(
                currentSteeringAngle,
                maxSteeringAngle,
                steeringSpeed * Time.deltaTime
            );
        }
        else
        {
            // Return to centre when no key held
            currentSteeringAngle = Mathf.MoveTowards(
                currentSteeringAngle, 0f,
                steeringSpeed * Time.deltaTime
            );
        }
    }

    bool IsKeyHeld(FinKey key)
    {
        switch (key)
        {
            case FinKey.Alpha1: return Keyboard.current.digit1Key.isPressed;
            case FinKey.Alpha2: return Keyboard.current.digit2Key.isPressed;
            case FinKey.Alpha3: return Keyboard.current.digit3Key.isPressed;
            case FinKey.Alpha4: return Keyboard.current.digit4Key.isPressed;
            case FinKey.Alpha5: return Keyboard.current.digit5Key.isPressed;
            case FinKey.Alpha6: return Keyboard.current.digit6Key.isPressed;
            case FinKey.Alpha7: return Keyboard.current.digit7Key.isPressed;
            case FinKey.Alpha8: return Keyboard.current.digit8Key.isPressed;
            default:            return false;
        }
    }

    void AnimateFin()
    {
        float targetDeployAngle = isDeployed ? deployedAngle : retractedAngle;

        currentDeployAngle = Mathf.MoveTowards(
            currentDeployAngle,
            targetDeployAngle,
            deploySpeed * Time.deltaTime
        );

        ApplyRotation();
    }

    void ApplyRotation()
    {
        // X axis = deploy rotation
        // Z axis = steering deflection
        transform.localRotation = Quaternion.Euler(
            currentDeployAngle,
            0f,
            currentSteeringAngle
        );
    }

    void ApplyAeroForce()
    {
        float altitude = Mathf.Max(0f, transform.position.y);
        float velocity = rb.linearVelocity.magnitude;

        if (velocity < 0.01f)
        {
            currentForce = 0f;
            currentEffectiveArea = 0f;
            return;
        }

        // Air density
        float airDensity = seaLevelAirDensity *
                           Mathf.Exp(-altitude / atmosphereScaleHeight);

        // Velocity direction
        Vector3 velocityDir = rb.linearVelocity.normalized;

        // Fin surface normal — local Y axis
        Vector3 finNormal = transform.up;

        // Effective area from angle between fin face and airflow
        float dotProduct = Mathf.Abs(Vector3.Dot(velocityDir, finNormal));
        currentEffectiveArea = finArea * dotProduct;

        // F = 0.5 * rho * v^2 * Cd * A_effective
        currentForce = 0.5f * airDensity
                       * velocity * velocity
                       * finDragCoefficient
                       * currentEffectiveArea;

        currentForce = Mathf.Min(currentForce, maxFinForce);

        // Base drag force opposing velocity
        Vector3 aeroForce = -velocityDir * currentForce;

        // Additional steering force from deflection angle
        // When fin is deflected it redirects airflow creating a lateral force
        if (Mathf.Abs(currentSteeringAngle) > 0.1f)
        {
            // Steering force acts along fin's local Z axis
            // proportional to deflection angle and dynamic pressure
            float dynamicPressure = 0.5f * airDensity * velocity * velocity;
            float steeringForce   = dynamicPressure
                                    * finArea
                                    * Mathf.Sin(currentSteeringAngle * Mathf.Deg2Rad)
                                    * finDragCoefficient;

            Vector3 steeringDir   = transform.right;
            aeroForce            += steeringDir * steeringForce;
        }

        // Apply at fin position — creates stabilising and steering torque
        rb.AddForceAtPosition(aeroForce, transform.position, ForceMode.Force);
    }

    void OnGUI()
    {
        float altitude     = Mathf.Max(0f, transform.position.y);
        string deployStatus = isDeployed ? "DEPLOYED" : "RETRACTED";
        string colour      = isDeployed ? "green" : "white";

        // Force display — show 0.000 instead of blank
        string forceStr = (currentForce / 1000f).ToString("F3") + " kN";

        // Stagger each fin display
        int finIndex = int.Parse(
            gameObject.name[gameObject.name.Length - 1].ToString()) - 1;
        float yPos = 430f + (finIndex * 20f);

        GUI.Box(new Rect(8, yPos, 320, 18), "");
        GUILayout.BeginArea(new Rect(14, yPos + 2, 308, 16));
        GUILayout.Label("<color=" + colour + ">"
                        + gameObject.name
                        + " : " + deployStatus
                        + " | Angle: " + currentDeployAngle.ToString("F0") + "°"
                        + " | Steer: " + currentSteeringAngle.ToString("F1") + "°"
                        + " | Force: " + forceStr
                        + "</color>");
        GUILayout.EndArea();
    }
}