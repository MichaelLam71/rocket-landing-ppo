using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.InputSystem.Controls;

public class PipeThruster : MonoBehaviour
{
    [Header("Thruster Settings")]
    [Tooltip("Force in Newtons.")]
    public float thrustForce = 250f;

    [Tooltip("Hold assigned key to fire. Release to stop.")]
    public bool isFiring = false;

    [Header("Key Assignment")]
    [Tooltip("Choose which key activates this thruster.")]
    public ThrusterKey assignedKey = ThrusterKey.T;

    [Header("Direction")]
    [Tooltip("Flip this if thrust fires the wrong way.")]
    public bool flipDirection = false;

    [Header("Atmospheric Effectiveness")]
    [Range(0f, 1f)]
    public float atmosphericDampingStrength = 0.8f;
    public float effectivenessAltitude = 40000f;
    public float seaLevelAirDensity = 1.225f;
    public float atmosphereScaleHeight = 8500f;

    [Header("Flame Effect")]
    public ParticleSystem thrusterFlame;

    // Enum for key selection in Inspector
    public enum ThrusterKey
    {
        T, Y, U, I, H, J, K, L
    }

    private Rigidbody rb;
    private float currentEffectiveness = 1f;

    void Start()
    {
        rb = GetComponentInParent<Rigidbody>();

        if (rb == null)
        {
            Debug.LogError("PipeThruster on " + gameObject.name +
                           ": No Rigidbody found in parent!");
        }

        if (thrusterFlame == null)
        {
            thrusterFlame = GetComponentInChildren<ParticleSystem>();
        }

        if (thrusterFlame != null)
        {
            thrusterFlame.Stop();
        }
        else
        {
            Debug.LogWarning("PipeThruster on " + gameObject.name +
                             ": No Particle System found in children!");
        }
    }

    void Update()
    {
        HandleInput();
    }

    void FixedUpdate()
    {
        if (isFiring && rb != null)
        {
            ApplyThrust();
        }
    }

    void HandleInput()
    {
        bool keyHeld = IsAssignedKeyHeld();

        if (keyHeld)
        {
            if (!isFiring)
            {
                isFiring = true;
                if (thrusterFlame != null)
                    thrusterFlame.Play();
                Debug.Log("PipeThruster " + gameObject.name + ": FIRING");
            }
        }
        else
        {
            if (isFiring)
            {
                isFiring = false;
                if (thrusterFlame != null)
                    thrusterFlame.Stop();
                Debug.Log("PipeThruster " + gameObject.name + ": OFF");
            }
        }
    }

    bool IsAssignedKeyHeld()
    {
        switch (assignedKey)
        {
            case ThrusterKey.T: return Keyboard.current.tKey.isPressed;
            case ThrusterKey.Y: return Keyboard.current.yKey.isPressed;
            case ThrusterKey.U: return Keyboard.current.uKey.isPressed;
            case ThrusterKey.I: return Keyboard.current.iKey.isPressed;
            case ThrusterKey.H: return Keyboard.current.hKey.isPressed;
            case ThrusterKey.J: return Keyboard.current.jKey.isPressed;
            case ThrusterKey.K: return Keyboard.current.kKey.isPressed;
            case ThrusterKey.L: return Keyboard.current.lKey.isPressed;
            default:            return false;
        }
    }

    void ApplyThrust()
    {
        Vector3 worldThrustDir;
        if (thrusterFlame != null)
        {
            worldThrustDir = flipDirection
                           ? thrusterFlame.transform.forward
                           : -thrusterFlame.transform.forward;
        }
        else
        {
            worldThrustDir = flipDirection ? -transform.up : transform.up;
        }

        float altitude       = Mathf.Max(0f, transform.position.y);
        float airDensity     = seaLevelAirDensity *
                               Mathf.Exp(-altitude / atmosphereScaleHeight);
        float maxDensity     = seaLevelAirDensity *
                               Mathf.Exp(-effectivenessAltitude / atmosphereScaleHeight);
        float densityRatio   = Mathf.Clamp01(airDensity / maxDensity);
        currentEffectiveness = Mathf.Lerp(1f, 1f - atmosphericDampingStrength,
                                          densityRatio);

        float effectiveForce = thrustForce * currentEffectiveness;
        Vector3 force        = worldThrustDir * effectiveForce;

        rb.AddForceAtPosition(force, transform.position, ForceMode.Force);
    }

    void OnGUI()
    {
        // Only show HUD for firing thrusters
        if (!isFiring) return;

        // Stagger vertical position based on thruster key
        // so multiple firing thrusters dont overlap
        int keyIndex = (int)assignedKey;
        float yPos = 370f + (keyIndex * 25f);

        string keyName  = assignedKey.ToString();
        string effStr   = (currentEffectiveness * 100f).ToString("F0") + "%";

        GUI.Box(new Rect(8, yPos, 280, 22), "");
        GUILayout.BeginArea(new Rect(14, yPos + 3, 268, 20));
        GUILayout.Label("<color=cyan>" + gameObject.name
                        + " [" + keyName + "] FIRING — "
                        + effStr + " effective</color>");
        GUILayout.EndArea();
    }
}