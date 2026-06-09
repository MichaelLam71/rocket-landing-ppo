using System;
using System.Net.Sockets;
using System.Threading;
using UnityEngine;

public class PythonBridge : MonoBehaviour
{
    private TcpClient client;
    private NetworkStream stream;
    private RocketController rocket;
    private Rigidbody rb;

    [Header("Connection")]
    public int port = 5005;

    [Header("Environment")]
    public Transform landingPad;
    public float maxEpisodeTime = 30f;
    public float timeScale = 10f;
    public bool useGimbal = true;

    [Header("Done Conditions")]
    public float padHeight = 5.2f;
    public float landingSpeedLimit = 3f;
    public float landingTiltLimit = 10f;
    public float landingRadius = 15f;
    public float tiltCrashLimit = 45f;
    public float outOfBoundsHeight = 400f;
    public float outOfBoundsXZ = 200f;

    [Header("Observation Scaling")]
    public float posScale = 50f;
    public float velScale = 20f;
    public float angVelScale = 10f;
    public float toLandScale = 50f;

    private float episodeTimer = 0f;
    private bool connected = false;
    private bool landed = false;
    private bool crashed = false;
    private int obsSize = 15;

    // cached action: applied EVERY physics frame so timeScale works correctly
    private float cachedThrust = 0f;
    private float cachedGimbalX = 0f;
    private float cachedGimbalZ = 0f;

    // threading
    private float[] latestAction = null;
    private bool actionReady = false;
    private readonly object lockObj = new object();

    void Start()
    {
        rocket = GetComponent<RocketController>();
        rb = GetComponent<Rigidbody>();

        if (rocket == null) { Debug.LogError("RocketController missing!"); enabled = false; return; }
        if (landingPad == null) { Debug.LogError("LandingPad not assigned!"); enabled = false; return; }

        Time.timeScale = timeScale;

        try
        {
            client = new TcpClient("127.0.0.1", port);
            stream = client.GetStream();
            connected = true;
            Debug.Log("Connected to Python!");

            Thread recvThread = new Thread(ReceiveLoop) { IsBackground = true };
            recvThread.Start();
        }
        catch (Exception e)
        {
            Debug.LogError("Could not connect: " + e.Message);
            enabled = false;
        }
    }

    void ReceiveLoop()
    {
        while (connected)
        {
            float[] action = RecvFloats(3);
            if (action == null) { connected = false; break; }

            lock (lockObj) { latestAction = action; actionReady = true; }

            while (connected)
            {
                lock (lockObj) { if (!actionReady) break; }
                Thread.Sleep(1);
            }
        }
    }

    void FixedUpdate()
    {
        // ALWAYS apply cached thrust every physics frame.
        // Between Python commands, the rocket keeps thrusting. Makes timeScale safe.
        rocket.ApplyAction(cachedThrust, cachedGimbalX, cachedGimbalZ);
        episodeTimer += Time.fixedDeltaTime;

        if (!connected) return;

        float[] action;
        lock (lockObj)
        {
            if (!actionReady) return;
            action = latestAction;
        }

        try
        {
            if (action[0] <= -990f)
            {
                // Reset
                rocket.ResetRocket();
                episodeTimer = 0f;
                landed = false;
                crashed = false;
                cachedThrust = 0f;
                cachedGimbalX = 0f;
                cachedGimbalZ = 0f;
                SendState(false);
            }
            else
            {
                // Apply action
                cachedThrust  = Mathf.Clamp01((action[0] + 1f) / 2f);
                cachedGimbalX = useGimbal ? Mathf.Clamp(action[1], -1f, 1f) : 0f;
                cachedGimbalZ = useGimbal ? Mathf.Clamp(action[2], -1f, 1f) : 0f;

                bool done = CheckDone();
                SendState(done);

                // Cut engine on episode end
                if (done)
                {
                    cachedThrust = 0f;
                    cachedGimbalX = 0f;
                    cachedGimbalZ = 0f;
                }
            }
        }
        catch (Exception e)
        {
            Debug.LogError("FixedUpdate error: " + e.Message);
            SendState(true);
        }

        lock (lockObj) { actionReady = false; }
    }

    // ==================== DONE CONDITIONS ====================
    bool CheckDone()
    {
        float height = transform.position.y;
        float speed  = rb.linearVelocity.magnitude;
        float tilt   = Vector3.Angle(transform.up, Vector3.up);

        landed = false;
        crashed = false;

        if (height <= padHeight)
        {
            float distToPad = Vector2.Distance(
                new Vector2(transform.position.x, transform.position.z),
                new Vector2(landingPad.position.x, landingPad.position.z));

            if (speed < landingSpeedLimit && tilt < landingTiltLimit && distToPad < landingRadius)
            {
                landed = true;
            }
            else
            {
                crashed = true;
            }
            return true;
        }
        if (tilt > tiltCrashLimit)         { crashed = true; return true; }
        if (episodeTimer > maxEpisodeTime) { crashed = true; return true; }
        if (height > outOfBoundsHeight ||
            Mathf.Abs(transform.position.x) > outOfBoundsXZ ||
            Mathf.Abs(transform.position.z) > outOfBoundsXZ)
        { crashed = true; return true; }

        return false;
    }

    // ==================== REWARD ====================
    float ComputeReward(bool done)
    {
        float speed = rb.linearVelocity.magnitude;

        if (landed)
        {
            float tilt = Vector3.Angle(transform.up, Vector3.up);
            float distToPad = Vector2.Distance(
                new Vector2(transform.position.x, transform.position.z),
                new Vector2(landingPad.position.x, landingPad.position.z));
            float softBonus = (1f - speed / landingSpeedLimit) * 100f;
            float tiltBonus = (1f - tilt / landingTiltLimit) * 50f;
            float proximityBonus = (1f - distToPad / landingRadius) * 50f;
            return 100f + softBonus + tiltBonus + proximityBonus;
        }
        if (crashed) return -100f;

        return -0.01f;
    }

    // ==================== OBSERVATION ====================
    float[] GetObservation()
    {
        Vector3 pos    = transform.position;
        Vector3 vel    = rb.linearVelocity;
        Vector3 angVel = rb.angularVelocity;
        Vector3 up     = transform.up;
        Vector3 toLand = landingPad.position - pos;

        float[] obs = new float[]
        {
            pos.x / posScale, pos.y / posScale, pos.z / posScale,
            vel.x / velScale, vel.y / velScale, vel.z / velScale,
            up.x, up.y, up.z,
            angVel.x / angVelScale, angVel.y / angVelScale, angVel.z / angVelScale,
            toLand.x / toLandScale, toLand.y / toLandScale, toLand.z / toLandScale
        };

        for (int i = 0; i < obs.Length; i++)
            obs[i] = Mathf.Clamp(obs[i], -5f, 5f);

        return obs;
    }

    // ==================== NETWORK ====================
    void SendState(bool done)
    {
        float[] obs = GetObservation();
        float reward = ComputeReward(done);

        float[] response = new float[obsSize + 2];
        Array.Copy(obs, response, obsSize);
        response[obsSize]     = reward;
        response[obsSize + 1] = done ? 1f : 0f;
        SendFloats(response);
    }

    void SendFloats(float[] values)
    {
        try
        {
            byte[] data = new byte[values.Length * 4];
            Buffer.BlockCopy(values, 0, data, 0, data.Length);
            stream.Write(data, 0, data.Length);
        }
        catch (Exception e) { Debug.LogError("Send: " + e.Message); connected = false; }
    }

    float[] RecvFloats(int count)
    {
        try
        {
            byte[] data = new byte[count * 4];
            int received = 0;
            while (received < data.Length)
            {
                int bytes = stream.Read(data, received, data.Length - received);
                if (bytes == 0) return null;
                received += bytes;
            }
            float[] values = new float[count];
            Buffer.BlockCopy(data, 0, values, 0, data.Length);
            return values;
        }
        catch { return null; }
    }

    void OnDestroy()
    {
        connected = false;
        Time.timeScale = 1f;
        stream?.Close();
        client?.Close();
    }
    
}